import http from 'k6/http';
import { baseUrl, checkForecast, payload, summary } from './lib.js';
export const options = { scenarios: { spike: { executor: 'ramping-arrival-rate', startRate: 5, timeUnit: '1s', preAllocatedVUs: 100, stages: [{ target: Number(__ENV.RATE || 200), duration: '5s' }, { target: Number(__ENV.RATE || 200), duration: '20s' }, { target: 5, duration: '5s' }] } } };
export default function () { checkForecast(http.post(`${baseUrl}/v2/forecasts`, payload(), { headers: { 'Content-Type': 'application/json' } })); }
export function handleSummary(data) { return summary(data); }
