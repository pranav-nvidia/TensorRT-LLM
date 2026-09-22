// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// Local response fixture only. It never replaces Blossom, AUTH, or the scan gate.
'use strict';
const http = require('node:http');

const scenarios = Object.freeze({
  'http-200': {status: 200, outcome: 'accepted'},
  'http-204': {status: 204, outcome: 'accepted'},
  'http-299': {status: 299, outcome: 'accepted'},
  'http-302': {status: 302, outcome: 'refused'},
  'http-403': {status: 403, outcome: 'refused'},
  'http-404': {status: 404, outcome: 'refused'},
  'http-500': {status: 500, outcome: 'refused'},
  'http-503': {status: 503, outcome: 'refused'},
  'body-at-limit': {status: 200, outcome: 'accepted', bytes: 16 * 1024},
  'body-over-limit': {status: 200, outcome: 'oversized', bytes: 16 * 1024 + 1},
  'body-truncated': {status: 200, outcome: 'unreadable', truncated: true},
  'connection-reset': {status: null, outcome: 'request-error', reset: true},
});

async function withWebhookFixture(name, exercise) {
  const scenario = scenarios[name];
  if (!scenario) throw new Error('Unknown webhook fixture scenario');
  const marker = 'PASS2_RESPONSE_BODY_' + name;
  const statusMarker = 'PASS2_STATUS_TEXT_' + name;
  const evidence = {scenario: name, configuredStatus: scenario.status, dispatchRequests: 0,
    redirectRequests: 0, unexpectedRequests: 0};
  const sockets = new Set();
  const server = http.createServer((req, res) => {
    // Drain without retaining or printing the payload or query: they may contain credentials.
    req.resume();
    const pathname = new URL(req.url, 'http://127.0.0.1').pathname;
    if (pathname === '/redirect') {
      evidence.redirectRequests++;
      res.writeHead(200);
      res.end(marker);
      return;
    }
    if (req.method !== 'POST' || pathname !== '/generic-webhook-trigger/invoke') {
      evidence.unexpectedRequests++;
      res.writeHead(400);
      res.end();
      return;
    }
    evidence.dispatchRequests++;
    if (scenario.reset) {
      req.socket.destroy();
      return;
    }
    res.statusCode = scenario.status;
    res.statusMessage = statusMarker;
    res.setHeader('Content-Type', 'text/plain');
    if (scenario.status === 302) res.setHeader('Location', '/redirect');
    if (scenario.truncated) {
      res.setHeader('Content-Length', marker.length + 1024);
      res.flushHeaders();
      res.write(marker);
      setTimeout(() => res.destroy(), 50);
      return;
    }
    const body = scenario.bytes ? marker.padEnd(scenario.bytes, 'x') : marker;
    res.end(body);
  });
  server.on('connection', socket => {
    sockets.add(socket);
    socket.on('close', () => sockets.delete(socket));
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const port = server.address().port;
  try {
    return await exercise({scenario, evidence, marker, statusMarker,
      ciServer: `http://127.0.0.1:${port}@pass2-local-only`,
      endpoint: `http://127.0.0.1:${port}/generic-webhook-trigger/invoke?token=pass2-local-only`});
  } finally {
    for (const socket of sockets) socket.destroy();
    await new Promise(resolve => server.close(resolve));
  }
}

function assessDispatch(fixture, observed) {
  // observed must come from the real child process + audit file + GitHub API queries.
  // This helper never invokes an endpoint or manufactures Blossom reactions/statuses.
  if (fixture.evidence.dispatchRequests === 0) {
    return {result: 'BLOCKED', reason: 'Blossom did not reach the controlled webhook'};
  }
  const failures = [];
  const requireCheck = (condition, message) => {if (!condition) failures.push(message);};
  requireCheck(fixture.evidence.dispatchRequests === 1, 'Expected one dispatch request');
  requireCheck(fixture.evidence.redirectRequests === 0, 'Redirect was followed');
  requireCheck(fixture.evidence.unexpectedRequests === 0, 'Unexpected endpoint request');
  const combined = [observed.stdout, observed.stderr, observed.auditText].join('\n');
  requireCheck(!combined.includes(fixture.marker), 'Response body leaked into output');
  requireCheck(!combined.includes(fixture.statusMarker), 'Server status text leaked into output');
  const accepted = fixture.scenario.outcome === 'accepted';
  requireCheck(observed.exit === (accepted ? 0 : 255), 'Unexpected operator exit');
  requireCheck(JSON.stringify(observed.newReactions) === JSON.stringify([accepted ? '+1' : '-1']),
    'Unexpected new comment reactions');
  const descriptions = {
    accepted: 'CI dispatch accepted !!!', refused: 'CI dispatch refused !!!',
    oversized: 'CI dispatch response too large !!!', unreadable: 'CI dispatch response could not be read !!!',
    'request-error': 'CI dispatch acceptance could not be confirmed !!!',
  };
  const status = observed.newStatuses || [];
  requireCheck(status.length === 1 && status[0].state === (accepted ? 'pending' : 'failure') &&
    status[0].description === descriptions[fixture.scenario.outcome] &&
    status[0].target_url === observed.runUrl, 'Missing or mismatched fresh commit status');
  requireCheck((observed.auditRecords || []).some(r => accepted ? r.status === 'success' :
    r.status === 'failure' && r.code === 'job-start'), 'Missing dispatch audit verdict');
  if (fixture.scenario.outcome === 'refused') {
    requireCheck(observed.stdout.includes(`CI server refused dispatch: HTTP ${fixture.scenario.status}`),
      'Missing numeric refusal diagnostic');
  }
  if (accepted) {
    requireCheck(observed.stdout.includes(`CI server accepted dispatch: HTTP ${fixture.scenario.status}`),
      'Missing numeric acceptance diagnostic');
  }
  return {result: failures.length ? 'FAIL' : 'PASS', failures};
}
module.exports = {scenarios, withWebhookFixture, assessDispatch};
