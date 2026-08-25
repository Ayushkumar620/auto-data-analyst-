import type {
  ForecastResult,
  ScenarioComparison,
  ScenarioResult,
} from '../types';
import { authedFetch, buildApiUrl } from './api';

export type RunForecastParams = {
  dataset: Array<Record<string, unknown>>;
  target_column?: string;
  time_column?: string;
  forecast_horizon?: number;
  confidence_level?: number;
};

export type RunWhatIfParams = {
  dataset: Array<Record<string, unknown>>;
  target: string;
  scenario_name?: string;
  changed_variables?: Record<string, unknown>;
  assumptions?: string[];
  scenarios?: Record<string, Record<string, unknown>>;
};

export async function runForecast(params: RunForecastParams): Promise<ForecastResult> {
  const res = await authedFetch(buildApiUrl('/api/v1/forecast/run'), {
    method: 'POST',
    body: JSON.stringify({
      dataset: params.dataset,
      target_column: params.target_column || null,
      time_column: params.time_column || null,
      forecast_horizon: params.forecast_horizon || 6,
      confidence_level: params.confidence_level || 0.8,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Forecasting failed to execute');
  }

  return res.json();
}

export async function runWhatIfScenario(params: RunWhatIfParams): Promise<ScenarioResult | ScenarioComparison> {
  const res = await authedFetch(buildApiUrl('/api/v1/forecast/whatif'), {
    method: 'POST',
    body: JSON.stringify({
      dataset: params.dataset,
      target: params.target,
      scenario_name: params.scenario_name || 'Custom Scenario',
      changed_variables: params.changed_variables || {},
      assumptions: params.assumptions || [],
      scenarios: params.scenarios || null,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'What-If scenario simulation failed');
  }

  return res.json();
}

