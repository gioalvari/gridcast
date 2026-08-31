# Data sources

## PJM hourly energy consumption

- Dataset: [Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
- Publisher: Rob Mulla
- Original source: [PJM Interconnection](https://www.pjm.com/)
- File used: `PJME_hourly.csv`
- Dataset version: 3
- Coverage: January 2002 through August 2018
- License: CC0 1.0 Public Domain

The source file contains hourly electricity-consumption estimates in megawatts.
GridCast does not redistribute the dataset. The ingestion command downloads it
from Kaggle and stores it under the ignored local `data/` directory.

The wall-clock timestamps contain daylight-saving irregularities. GridCast
records these in `artifacts/data/pjme_quality.json`, averages duplicate
timestamps, and interpolates only short missing intervals before creating the
model-ready Parquet file. The normalized timestamps intentionally remain naive
because the republished file does not provide an explicit timezone or UTC
offset with which to disambiguate every historical transition.

## Philadelphia hourly temperature

- API: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- Upstream dataset: ERA5 reanalysis
- Location: Philadelphia, Pennsylvania (39.9526, -75.1652)
- Variable: 2-metre air temperature in degrees Celsius
- Coverage requested: January 2002 through August 2018
- Resolution: hourly

Philadelphia is used as a transparent weather proxy for the PJM East load
series; a single point cannot represent the complete PJME footprint. ERA5 is
reanalysis, not an archived day-ahead forecast. To avoid hindsight leakage,
GridCast never uses contemporaneous realized temperature in forecast rows. It
uses temperatures delayed by one and two weeks and a climatology computed from
prior years only. Historical operational forecasts do not cover the complete
2002-2018 evaluation period.

## ENTSO-E Transparency Platform

- API: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)
- Dataset: actual total load (`documentType=A65`, `processType=A16`)
- Default area: Italy bidding zone (`10YIT-GRTN-----B`)
- Authentication: personal ENTSO-E security token
- Output timezone: UTC
- License and reuse: subject to the ENTSO-E Transparency Platform terms

GridCast retrieves monthly XML chunks, supports 15-, 30-, and 60-minute period
resolutions, validates temporal completeness, and caches normalized data as
Snappy-compressed Parquet. Authenticated raw XML, processed datasets, and tokens
remain outside Git.

The token is read from `ENTSOE_API_TOKEN` in CI or from the macOS Keychain
service `gridcast-entsoe-api-token` locally. The authenticated URL is passed to
curl through standard input rather than process arguments, so the token does
not appear in process listings or command logs.

## Open-Meteo archived model runs

- API: [Open-Meteo Single Runs API](https://open-meteo.com/en/docs/single-runs-api)
- Model: one explicit ECMWF IFS run initialized at 00 UTC on D-1
- Nodes: Bari, Bologna, Cagliari, Milan, Naples, Palermo, Rome, and Turin
- Variables: temperature, humidity, shortwave radiation, wind speed, cloud cover

Unlike ERA5 reanalysis, archived model runs represent forecasts from one model
vintage. GridCast records run initialization and conservative availability
timestamps, then rejects any run that was not public at the operational forecast
origin. Source availability does not imply unrestricted redistribution rights;
raw responses remain outside Git.
