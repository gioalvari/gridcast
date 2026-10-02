import http from 'k6/http';
import { baseUrl, checkForecast, payload, summary } from './lib.js';
export const options = { scenarios: { smoke: { executor: 'constant-arrival-rate', rate: Number(__ENV.RATE || 5), timeUnit: '1s', duration: __ENV.DURATION || '20s', preAllocatedVUs: 10 } }, thresholds: { http_req_failed: ['rate<0.01'], http_req_duration: ['p(95)<500'], checks: ['rate>0.99'] } };
export default function () { checkForecast(http.post(`${baseUrl}/v2/forecasts`, payload(), { headers: { 'Content-Type': 'application/json' } })); }
export function handleSummary(data) { return summary(data); }
