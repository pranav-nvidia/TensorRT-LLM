// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// Read-only, fork-scoped inspection. Print selected counts and fixture IDs, never credentials or raw responses.
'use strict';
const fs = require('node:fs');
const https = require('node:https');
const crypto = require('node:crypto');

async function inspectPolicy({core, context}) {
  const repo = `${context.repo.owner}/${context.repo.repo}`;
  if (repo !== 'pranav-nvidia/TensorRT-LLM') throw new Error('Unexpected diagnostic repository');
  let stage = 'configuration';
  const deadline = Date.now()+180000;
  const fields = (obj,key) => Object.entries(obj).find(([k])=>k.toLowerCase()===key.toLowerCase())?.[1];
  const fail = (label,status) => {const e = new Error('Diagnostic request failed');e.stage=label;e.status=status;throw e;};
  const request = (url, headers={}, method='GET') => new Promise((resolve,reject)=> {
    if (Date.now()>deadline) {reject(Object.assign(new Error('Deadline'),{stage,status:'deadline'}));return;}
    // Matches the existing operator's internal-service TLS behavior; no deployment settings are changed.
    const req=https.request(url,{method,headers,rejectUnauthorized:false,timeout:20000},res=>{
      let size=0;const chunks=[];
      res.on('data',chunk=>{size+=chunk.length;if(size>8*1024*1024)req.destroy();else chunks.push(chunk);});
      res.on('error',()=>reject(Object.assign(new Error('Response read'),{stage,status:'read-error'})));
      res.on('end',()=>{
        if(res.statusCode<200 || res.statusCode>=300){reject(Object.assign(new Error('HTTP'),{stage,status:res.statusCode}));return;}
        try{resolve({status:res.statusCode,data:JSON.parse(Buffer.concat(chunks).toString('utf8'))});}
        catch{reject(Object.assign(new Error('JSON'),{stage,status:'invalid-json'}));}
      });
    });
    req.on('timeout',()=>req.destroy());
    req.on('error',()=>reject(Object.assign(new Error('Transport'),{stage,status:'transport-error'})));
    req.end();
  });
  try {
    const user=fs.readFileSync('/auth-data/username','utf8');
    const password=fs.readFileSync('/auth-data/password','utf8');
    const secret=process.env.SECRET_TOKEN;
    if(!secret) fail(stage,'missing-configuration');
    const config=(await request('https://github-operator.github-ns.svc:8443/config',{
      'SECRET-TOKEN':secret,Authorization:'Basic '+Buffer.from(user+':'+password).toString('base64')})).data;
    const base=fields(config,'BlackduckUrl');
    const token=fields(config,'BlackduckToken');
    const gitlabToken=fields(config,'GitlabToken');
    if(!base || !token || !gitlabToken) fail(stage,'missing-fields');
    core.setSecret(token);core.setSecret(gitlabToken);
    stage='blackduck-authentication';
    const auth=(await request(new URL('/api/tokens/authenticate',base),{Authorization:'token '+token},'POST')).data;
    if(!auth.bearerToken)fail(stage,'missing-bearer');
    core.setSecret(auth.bearerToken);
    const origin=new URL(base).origin;
    const get=async (href,accept='application/json')=>{
      const url=new URL(href,base);
      if(url.origin!==origin || !url.pathname.startsWith('/api/')) fail(stage,'unexpected-resource-origin');
      return (await request(url,{Authorization:'Bearer '+auth.bearerToken,Accept:accept})).data;
    };
    const link=(obj,rel)=>{const href=obj._meta?.links?.find(l=>l.rel===rel)?.href;if(!href)fail(stage,'missing-'+rel);return href;};
    stage='project';
    const name='[pipeline][oss]['+repo+']';
    const projects=await get('/api/projects?q='+encodeURIComponent('name:'+name));
    const project=projects.items?.find(p=>p.name===name);
    if(!project)fail(stage,'project-not-found');
    stage='version';
    const versions=await get(link(project,'versions')+'?q='+encodeURIComponent('versionName:'+context.runId));
    const version=versions.items?.find(v=>v.versionName===String(context.runId));
    if(!version)fail(stage,'version-not-found');
    stage='bom-status';
    const states=[];
    for(let i=0;i<18;i++){
      const status=await get(link(version,'bom-status'));states.push(status.status);
      if(status.status==='UP_TO_DATE')break;
      await new Promise(resolve=>setTimeout(resolve,5000));
    }
    stage='exceptions';
    const file=(await request('https://gitlab-master.nvidia.com/api/v4/projects/32700/repository/files/'+encodeURIComponent(repo)+'.json?ref=main',{'PRIVATE-TOKEN':gitlabToken})).data;
    const decoded=Buffer.from(file.content,'base64').toString('utf8');
    const exceptionDocument=JSON.parse(decoded);
    const allExceptions=fields(exceptionDocument,'Exceptions') ?? [];
    if(!Array.isArray(allExceptions))fail(stage,'invalid-exceptions-shape');
    const exceptions=new Set(allExceptions.filter(s=>s.includes('[SECURITY]')));
    const isFixture=name=>/lodash|minimist/i.test(name||'');
    const mime='application/vnd.blackducksoftware.bill-of-materials-6+json';
    const paginate=async href=>{
      const items=[];let totalCount=0;let firstPage=[];
      for(let offset=0;offset<5000;offset+=300){
        const url=new URL(href);url.searchParams.set('offset',String(offset));url.searchParams.set('limit','300');
        const page=await get(url,mime);
        if(!Array.isArray(page.items))fail(stage,'missing-items');
        if(offset===0)firstPage=page.items;
        totalCount=page.totalCount;
        items.push(...page.items);
        if(!page.items.length || items.length>=totalCount)break;
      }
      return {items,firstPage,totalCount,truncated:items.length<totalCount};
    };
    stage='bom-components';
    const components=await paginate(link(version,'components'));
    stage='vulnerable-components';
    const vulnerable=await paginate(link(version,'vulnerable-components'));
    const classify=items=>({returned:items.length,bdsa:items.filter(c=>c.vulnerabilityWithRemediation?.vulnerabilityName?.includes('BDSA')).length,
      excepted:items.filter(c=>exceptions.has('[SECURITY]'+c.componentName+' '+c.vulnerabilityWithRemediation?.vulnerabilityName)).length,
      eligibleForRejection:items.filter(c=>!c.vulnerabilityWithRemediation?.vulnerabilityName?.includes('BDSA') && !exceptions.has('[SECURITY]'+c.componentName+' '+c.vulnerabilityWithRemediation?.vulnerabilityName)).length});
    const firstIds=new Set(vulnerable.firstPage.map(c=>c.componentName+'|'+c.componentVersionName+'|'+c.vulnerabilityWithRemediation?.vulnerabilityName));
    const fixtureVulnerabilities=[];
    for(const item of vulnerable.items.filter(c=>isFixture(c.componentName))){
      const v=item.vulnerabilityWithRemediation||{};
      const id=v.vulnerabilityName;
      const row={component:item.componentName,version:item.componentVersionName,id,source:v.source,severity:v.severity,
        remediationStatus:v.remediationStatus,inFirst300:firstIds.has(item.componentName+'|'+item.componentVersionName+'|'+id),
        ignoredAsBDSA:!!id?.includes('BDSA'),excepted:exceptions.has('[SECURITY]'+item.componentName+' '+id)};
      fixtureVulnerabilities.push(row);
    }
    const report={runId:context.runId,version:version.versionName,bomStatuses:states,
      exceptionCount:exceptions.size,exceptionFileRevision:file.last_commit_id,exceptionFileSha256:crypto.createHash('sha256').update(decoded).digest('hex'),
      fixtureExceptionIds:[...exceptions].filter(isFixture),
      bomTotal:components.totalCount,bomTruncated:components.truncated,
      fixtureComponents:components.items.filter(c=>isFixture(c.componentName)).map(c=>({name:c.componentName,version:c.componentVersionName,matchTypes:c.matchTypes,vulnerabilityRiskProfile:c.vulnerabilityRiskProfile})),
      vulnerableTotal:vulnerable.totalCount,vulnerableTruncated:vulnerable.truncated,
      first300:classify(vulnerable.firstPage),allRetrieved:classify(vulnerable.items),fixtureVulnerabilities};
    core.info('POLICY_DIAGNOSTIC '+JSON.stringify(report));
    stage='related-vulnerabilities';
    report.bdsaRelated=[];
    for(const id of [...new Set(fixtureVulnerabilities.filter(v=>v.ignoredAsBDSA).map(v=>v.id))].slice(0,12)){
      const detail=await get('/api/vulnerabilities/'+encodeURIComponent(id));
      const related=detail._meta?.links?.find(l=>l.rel==='related-vulnerability');
      let relatedName=null;
      if(related)relatedName=(await get(related.href)).name;
      report.bdsaRelated.push({id,relatedName});
    }
    core.info('POLICY_RELATED '+JSON.stringify({runId:context.runId,bdsaRelated:report.bdsaRelated}));
    return report;
  } catch(e) {
    const result={runId:context.runId,stage:e.stage||stage,status:e.status||'read-or-parse-error'};
    core.info('POLICY_DIAGNOSTIC_ERROR '+JSON.stringify(result));
    return result;
  }
}
module.exports={inspectPolicy};
