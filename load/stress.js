import http from 'k6/http';
import { baseUrl, checkForecast, payload, summary } from './lib.js';
export const options = { scenarios: { stress: { executor: 'ramping-arrival-rate', startRate: 20, timeUnit: '1s', preAllocatedVUs: 100, maxVUs: 500, stages: [{ target: 50, duration: '30s' }, { target: 100, duration: '30s' }, { target: 200, duration: '30s' }, { target: 300, duration: '30s' }] } } };
export default function () { checkForecast(http.post(`${baseUrl}/v2/forecasts`, payload(), { headers: { 'Content-Type': 'application/json' } })); }
export function handleSummary(data) { return summary(data); }
