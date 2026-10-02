import http from 'k6/http';
import { baseUrl, checkForecast, payload, summary } from './lib.js';
export const options = { scenarios: { soak: { executor: 'constant-arrival-rate', rate: Number(__ENV.RATE || 30), timeUnit: '1s', duration: __ENV.DURATION || '10m', preAllocatedVUs: 50 } } };
export default function () { checkForecast(http.post(`${baseUrl}/v2/forecasts`, payload(), { headers: { 'Content-Type': 'application/json' } })); }
export function handleSummary(data) { return summary(data); }
