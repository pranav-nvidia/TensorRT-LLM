// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// Invoke from actions/github-script in START-CI-JOB after a fresh real scan.
// Requires webhook-fixture.cjs alongside this module. The deployed-binary harness has run on the test runner.
'use strict';
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const cp = require('node:child_process');
const {withWebhookFixture, assessDispatch} = require('./webhook-fixture.cjs');
const consumedRuns = new Set();

async function runDispatchProbe({github, context, core, fixtureName, prNumber, expectedPrHeadSha}) {
  if (`${context.repo.owner}/${context.repo.repo}` !== 'pranav-nvidia/TensorRT-LLM') {
    throw new Error('This test preparation is scoped to the isolated fork');
  }
  if (process.env.GITHUB_RUN_ATTEMPT !== '1' || consumedRuns.has(context.runId)) {
    throw new Error('Use a new workflow run and fresh scan for each response fixture');
  }
  const blocked = reason => {
    const result = {case: fixtureName, result: 'BLOCKED', reason, runId: context.runId};
    core.info('PASS2_RESULT '+JSON.stringify(result));
    core.setFailed(`${fixtureName === 'policy-rejection' ? 'E06' : 'E08'} ${fixtureName}: BLOCKED; ${reason}`);
    return result;
  };
  const caseId = fixtureName === 'policy-rejection' ? 'E06' : 'E08';
  const token = process.env.PROBE_TOKEN;
  if (!token || !expectedPrHeadSha || !Number.isInteger(prNumber)) {
    throw new Error('Missing test credentials or captured PR identity');
  }
  const {data: jobs} = await github.rest.actions.listJobsForWorkflowRun({
    ...context.repo, run_id: context.runId, per_page: 100});
  if (!jobs.jobs.some(j => j.name === 'Vulnerability scan' && j.conclusion === 'success')) {
    return blocked('No successful real Vulnerability scan job in this run');
  }
  const {data: pr} = await github.rest.pulls.get({...context.repo, pull_number: prNumber});
  if (pr.state !== 'open' || pr.head.sha !== expectedPrHeadSha) {
    return blocked('PR closed or head changed since scan setup');
  }
  const policyTest = fixtureName === 'policy-rejection';
  const beforeComments = policyTest ? new Set((await github.paginate(github.rest.issues.listComments, {...context.repo, issue_number:prNumber, per_page:100})).map(c=>c.id)) : null;
  consumedRuns.add(context.runId);
  const {data: comment} = await github.rest.issues.createComment({...context.repo, issue_number: prNumber,
    body: `${caseId} pass 2 controlled webhook: ${fixtureName}. START-CI-JOB probe; this marker is not an AUTH receipt test.`});
  const statuses = () => github.paginate(github.rest.repos.listCommitStatusesForRef,
    {...context.repo, ref: expectedPrHeadSha, per_page: 100});
  const before = new Set((await statuses()).map(s => s.id));
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'blossom-pass2-'));
  const eventPath = path.join(dir, 'event.json');
  const auditPath = path.join(dir, 'audit.log');
  fs.writeFileSync(eventPath, JSON.stringify({action: 'created', comment: {id: comment.id, body: '/bot run'},
    issue: {number: prNumber, pull_request: {}}}));
  const binary = '/home/github/bin/blossom-ci';
  const binarySha256 = crypto.createHash('sha256').update(fs.readFileSync(binary)).digest('hex');
  const runUrl = `https://github.com/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`;
  try {
    return await withWebhookFixture(policyTest ? 'http-200' : fixtureName, async fixture => {
      // Preserve the real actor, configuration and run ID. Only event context, audit destination
      // and CI endpoint are supplied by this harness. AUTH and the vulnerability gate are not mocked.
      const env = {...process.env, OPERATION: 'START-CI-JOB', REPO_TOKEN: token,
        GITHUB_EVENT_NAME: 'issue_comment', GITHUB_EVENT_PATH: eventPath,
        AUDIT_LOG_FILE: auditPath, CI_SERVER: fixture.ciServer};
      const execution = await new Promise(resolve => cp.execFile(binary, [], {env, cwd: dir, timeout: 180000},
        (err, stdout, stderr) => resolve({exit: err ? err.code : 0, stdout, stderr})));
      const auditText = fs.existsSync(auditPath) ? fs.readFileSync(auditPath, 'utf8') : execution.stderr;
      const auditRecords = [];
      for (const line of auditText.split('\n')) {
        const start = line.indexOf('{');
        if (start < 0) continue;
        try {auditRecords.push(JSON.parse(line.slice(start)));} catch { /* Missing valid verdict fails assessment. */ }
      }
      const {data: reactions} = await github.rest.reactions.listForIssueComment({...context.repo, comment_id: comment.id});
      const newReactions = reactions.filter(r => r.user.login === 'github-actions[bot]').map(r => r.content);
      const added = (await statuses()).filter(s => !before.has(s.id) && s.context === 'blossom-ci' && s.target_url === runUrl);
      const newStatuses = added.map(s => ({state: s.state, description: s.description, target_url: s.target_url}));
      let verdict;
      let policyReports = [];
      if (policyTest) {
        const comments = await github.paginate(github.rest.issues.listComments,
          {...context.repo,issue_number:prNumber,per_page:100});
        policyReports = comments.filter(c=>!beforeComments.has(c.id) &&
          c.user.login==='github-actions[bot]' && c.body.includes('Promotion blocked, new vulnerability found') &&
          /lodash\s*\|\s*CVE-\d{4}-\d+/i.test(c.body)).map(c=>({id:c.id,url:c.html_url,
            rows:c.body.split('\n').filter(l=>/lodash\s*\|\s*CVE-\d{4}-\d+/i.test(l))}));
        const failures = [];
        if (fixture.evidence.dispatchRequests!==0) failures.push('Gate did not block dispatch');
        if (execution.exit!==255) failures.push('Expected exit 255');
        if (JSON.stringify(newReactions)!==JSON.stringify(['-1'])) failures.push('Missing failure reaction');
        if (newStatuses.length!==1 || newStatuses[0].state!=='failure' ||
            newStatuses[0].description!=='L2 vulnerability scan check failed !!!') failures.push('Missing fresh scan failure status');
        if (!auditRecords.some(r=>r.status==='failure' && r.code==='scan')) failures.push('Missing scan failure audit');
        if (auditRecords.some(r=>r.status==='job-start' || r.status==='success')) failures.push('Unexpected dispatch audit');
        if (!policyReports.length) failures.push('No new report with a concrete lodash CVE');
        if (execution.stdout.includes('Failed to get security vulnerability exceptions issue from gitlab')) failures.push('Exceptions loading failed');
        verdict = {result:failures.length?'FAIL':'PASS',failures};
      } else {
        verdict = assessDispatch(fixture, {...execution, auditText, auditRecords, newReactions, newStatuses, runUrl});
      }
      const {data: afterPr} = await github.rest.pulls.get({...context.repo, pull_number: prNumber});
      if (afterPr.head.sha !== expectedPrHeadSha) {
        verdict.result = 'INVALID'; verdict.reason = 'PR head changed during the probe';
      }
      // Publish selected evidence only. Never log the request body, URL token, full stdout or stderr.
      const diagnostics = execution.stdout.split('\n').filter(l => /^(CI server (accepted|refused) dispatch: HTTP \d+$|CI dispatch response exceeded the response size limit$|Failed to read CI dispatch response body$|Failed to dispatch CI job: HTTP request failed$|Failed to get security vulnerability exceptions issue from gitlab$|L2 vulnerability scan check failed !!!$)/.test(l));
      const result = {case: fixtureName, ...verdict, ...fixture.evidence, exit: execution.exit,
        binarySha256, comment: comment.html_url, runUrl, newReactions, newStatuses,
        auditVerdicts: auditRecords.map(r => ({status: r.status, code: r.code})), diagnostics, policyReports, expectedComponent: policyTest ? 'lodash' : undefined, expectedVersion: policyTest ? '4.17.11' : undefined};
      core.info('PASS2_RESULT '+JSON.stringify(result));
      if (result.result !== 'PASS') core.setFailed(`${caseId} ${fixtureName}: ${result.result}; inspect PASS2_RESULT`);
      return result;
    });
  } finally {
    fs.rmSync(dir, {recursive: true, force: true});
  }
}
module.exports = {runDispatchProbe};
