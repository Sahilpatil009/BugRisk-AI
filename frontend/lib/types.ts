export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
export type AnalysisStatus = "QUEUED" | "ANALYZING" | "PREDICTING" | "COMPLETED" | "FAILED";

export interface Repository {
  id: string;
  github_repo_id: string;
  name: string;
  owner: string;
  url: string;
  default_branch: string;
  is_private: boolean;
  created_at: string;
}

export type GitHubRepository = Omit<Repository, "id" | "created_at">;

export interface Analysis {
  id: string;
  repository_id: string;
  commit_sha: string | null;
  status: AnalysisStatus;
  change_risk_probability: number | null;
  overall_priority_score: number | null;
  risk_level: RiskLevel | null;
  model_version: string | null;
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface Explanation {
  feature_name: string;
  feature_value: number;
  shap_value: number;
}

export interface FileResult {
  id: string;
  analysis_id: string;
  file_path: string;
  file_priority_score: number;
  risk_level: RiskLevel;
  loc: number;
  complexity: number;
  code_churn: number;
  commit_count: number;
  contributor_count: number;
  file_age_days: number;
  lines_added: number;
  lines_deleted: number;
  dependency_count: number;
  explanations: Explanation[];
  recommendations: string[];
}

export interface ModelMetrics {
  model_version: string;
  model_name: string;
  trained: boolean;
  metrics: Record<string, number>;
  feature_names: string[];
  note?: string;
}
