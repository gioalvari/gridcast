import { check } from 'k6';

export const baseUrl = __ENV.BASE_URL || 'http://envoy:8080';

export function payload() {
  const history = [];
  const end = new Date('2025-01-01T00:00:00Z');
  for (let index = 336; index > 0; index -= 1) {
    const timestamp = new Date(end.getTime() - index * 3600 * 1000);
    history.push({ timestamp: timestamp.toISOString(), load_mw: 1000 + index });
  }
  return JSON.stringify({ history, horizon_hours: 168 });
}

export function checkForecast(response) {
  return check(response, {
    'status is 200': (item) => item.status === 200,
    'response schema has 168 ordered intervals': (item) => {
      try {
        const body = item.json();
        return body.forecasts.length === 168 && body.forecasts.every(
          (point) => point.p10_mw <= point.p50_mw && point.p50_mw <= point.p90_mw,
        );
      } catch (_) {
        return false;
      }
    },
  });
}

export function summary(data) {
  return { [`/results/${__ENV.RESULT_NAME || 'summary'}.json`]: JSON.stringify(data, null, 2) };
}
