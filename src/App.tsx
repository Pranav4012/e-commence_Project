import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, AlertCircle, ArrowUpRight, BarChart3, Bell, Boxes,
  BrainCircuit, Check, ChevronDown, Clock3, Database, FileCheck2,
  FileSpreadsheet, Gauge, LayoutDashboard, MapPin, Menu, PackageCheck,
  Search, Settings2, ShieldCheck, Sparkles, UploadCloud, X, Zap, Sun, Moon
} from "lucide-react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend,
  Line, ResponsiveContainer, Tooltip, XAxis, YAxis
} from "recharts";

type Audit = {
  filename: string; rows: number; columns: number; duplicate_rows: number;
  duplicate_pct: number; total_missing: number; missing_pct: number; memory_mb: number;
  date_candidates: { column: string; valid_pct: number; min: string | null; max: string | null }[];
  columns_profile: { name: string; dtype: string; missing: number; missing_pct: number; unique: number; examples: unknown[] }[];
  numeric_summary: { name: string; mean: number | null; median: number | null; min: number | null; max: number | null }[];
  preview: Record<string, unknown>[]; warnings: string[];
};
type Analytics = { time_series: { period: string; orders: number }[]; date_column_used: string | null; message: string };
type ModelResult = {
  target_column: string;
  features: string[];
  classes: string[];
  feature_schema: { name: string; kind: "number" | "category"; examples: string[] }[];
  split_method: string;
  rows_used: number;
  train_rows: number;
  test_rows: number;
  dropped_missing_target: number;
  dropped_missing_date: number;
  metrics: { accuracy: number; precision_macro: number; recall_macro: number; f1_macro: number; roc_auc: number | null; roc_auc_positive_class: string | null };
  class_metrics: { label: string; precision: number; recall: number; f1: number; support: number }[];
  confusion_matrix: number[][];
  sample_predictions: { row: number; actual: string; predicted: string; max_class_probability: number }[];
  model_name: string;
  feature_importance: { feature: string; importance: number }[];
  feature_importance_method: string;
  robustness: {
    method: string;
    sample_rows: number;
    baseline_accuracy: number;
    baseline_balanced_accuracy: number;
    tests: { feature: string; accuracy: number; balanced_accuracy: number; balanced_accuracy_delta: number; prediction_flip_rate: number }[];
  };
};
type PredictionResult = {
  target_column: string;
  predicted_class: string;
  max_class_probability: number;
  probabilities: { label: string; probability: number }[];
};
type ExplanationResult = {
  target_column: string;
  predicted_class: string;
  predicted_probability: number;
  baseline: string;
  method: string;
  features: { feature: string; value: string; baseline_value: string; probability_at_value: number; probability_at_baseline: number; delta: number }[];
};
type TrackingAnalytics = {
  total_orders: number;
  completed_orders: number;
  open_orders: number;
  delayed_completed: number;
  delay_rate: number | null;
  average_late_days: number | null;
  status_completion_conflicts: number;
  delivery_dates_on_open: number;
  order_date_range: { start: string | null; end: string | null };
  monthly: { month: string; orders: number; completed: number; delayed: number; delay_rate: number | null }[];
  statuses: { status: string; count: number }[];
  summary: string;
  source: string;
};
type TrackedOrder = {
  order_id: string;
  order_date: string | null;
  status: string;
  completed: boolean;
  delayed: boolean | null;
  product: string | null;
  destination: string | null;
  shipping_service: string | null;
};
type OrderPage = { items: TrackedOrder[]; total: number; page: number; page_size: number; statuses: string[] };
type OrderDetail = {
  order_id: string;
  status: string;
  completed: boolean;
  is_delayed: boolean | null;
  delay_flag_recorded: boolean | null;
  order_date: string | null;
  actual_delivery_date: string | null;
  promised_delivery_days: number | null;
  promised_arrival_date: string | null;
  actual_delivery_days: number | null;
  delay_days: number | null;
  estimated_arrival_date: string | null;
  estimated_transit_days: number | null;
  estimated_remaining_days: number | null;
  eta_overdue_days: number | null;
  probability_by_promised_date: number | null;
  probability_interval: { lower: number; upper: number } | null;
  probability_label: string;
  arrival_probabilities: { transit_day: number; probability: number }[];
  cohort_count: number;
  cohort_description: string | null;
  product: string | null;
  category: string | null;
  destination: string | null;
  shipping_service: string | null;
  fulfilment: string | null;
  possible_contributors: { factor: string; detail: string }[];
  data_notes: string[];
  timeline: { date: string | null; label: string; state: string }[];
  source: string;
};

const nav = [
  { label: "Workspace", items: [
    { id: "overview", name: "Overview", icon: LayoutDashboard },
    { id: "analytics", name: "Analytics", icon: BarChart3 },
    { id: "predict", name: "Order prediction", icon: Zap },
    { id: "tracking", name: "Package tracking", icon: MapPin }
  ]},
  { label: "Intelligence", items: [
    { id: "models", name: "Model performance", icon: Gauge },
    { id: "explain", name: "Explainability", icon: BrainCircuit },
    { id: "robustness", name: "Robustness", icon: ShieldCheck }
  ]},
  { label: "System", items: [
    { id: "data", name: "Data quality", icon: Database }
  ]}
];

function App() {
  const [page, setPage] = useState("overview");
  const [audit, setAudit] = useState<Audit | null>(null);
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [model, setModel] = useState<ModelResult | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResult | null>(null);
  const [trackingAnalytics, setTrackingAnalytics] = useState<TrackingAnalytics | null>(null);
  const [trackingAnalyticsError, setTrackingAnalyticsError] = useState("");
  const [training, setTraining] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [restoring, setRestoring] = useState(true);
  const [error, setError] = useState("");
  const [mobileNav, setMobileNav] = useState(false);
  const [search, setSearch] = useState("");
  const [theme, setTheme] = useState<"dark" | "light">(() => localStorage.getItem("delivery-intelligence-theme") === "light" ? "light" : "dark");
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.body.dataset.theme = theme;
    localStorage.setItem("delivery-intelligence-theme", theme);
  }, [theme]);

  useEffect(() => {
    let active=true;
    fetch("/api/audit/current").then(async response=>{
      if(response.status===404)return null;
      const data=await response.json();
      if(!response.ok)throw new Error(data.detail||"Could not restore the uploaded dataset.");
      return data as Audit;
    }).then(data=>{if(active&&data)setAudit(data);}).catch(error=>{if(active)setError(error instanceof Error?error.message:"Could not connect to the dataset API.");}).finally(()=>{if(active)setRestoring(false);});
    fetch("/api/model/current").then(async response=>{
      if(response.status===404)return null;
      const data=await response.json();
      if(!response.ok)throw new Error(data.detail||"Could not restore the trained model.");
      return data as ModelResult;
    }).then(data=>{if(active&&data)setModel(data);}).catch(error=>{if(active)setError(error instanceof Error?error.message:"Could not connect to the model API.");});
    return ()=>{active=false;};
  }, []);

  useEffect(() => {
    if (audit) fetch("/api/analytics").then(r => r.ok ? r.json() : null).then(d => d && setAnalytics(d)).catch(() => {});
  }, [audit]);

  useEffect(() => {
    if (!audit) return;
    let active=true;
    setTrackingAnalytics(null);
    setTrackingAnalyticsError("");
    requestJson<TrackingAnalytics>("/api/tracking/analytics")
      .then(data=>{if(active)setTrackingAnalytics(data);})
      .catch(error=>{if(active)setTrackingAnalyticsError(error instanceof Error?error.message:"Delivery analytics are unavailable for this CSV.");});
    return ()=>{active=false;};
  }, [audit]);

  async function handleFile(file?: File) {
    if (!file) return;
    setError("");
    setUploading(true);
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch("/api/audit", { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not audit this file.");
      setAudit(data);
      setModel(null);
      setPrediction(null);
      setExplanation(null);
      setTrackingAnalytics(null);
      setPage("data");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  async function handleTrain(target: string, features: string[], dateColumn: string) {
    setError("");
    setTraining(true);
    setModel(null);
    setPrediction(null);
    setExplanation(null);
    try {
      const response = await fetch("/api/model/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_column: target,
          feature_columns: features,
          date_column: dateColumn || null
        })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Model training failed.");
      setModel(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Model training failed.");
    } finally {
      setTraining(false);
    }
  }

  async function handlePredict(values: Record<string, string>) {
    setError("");
    setPrediction(null);
    try {
      const response = await fetch("/api/model/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: values })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not generate a prediction.");
      setPrediction(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not generate a prediction.");
    }
  }

  async function handleExplain(values: Record<string, string>) {
    setError("");
    setExplanation(null);
    try {
      const response = await fetch("/api/model/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: values })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not explain this prediction.");
      setExplanation(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not explain this prediction.");
    }
  }

  const filteredColumns = useMemo(() => {
    if (!audit) return [];
    return audit.columns_profile.filter(c => c.name.toLowerCase().includes(search.toLowerCase()));
  }, [audit, search]);

  const activeTitle = nav.flatMap(s => s.items).find(i => i.id === page)?.name || "Overview";

  return (
    <div className={`app-shell theme-${theme}`}>
      <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><PackageCheck size={19} /></div>
          <div><div className="brand-name">delivery<span>intel</span></div><div className="brand-caption">RESEARCH WORKSPACE</div></div>
          <button className="icon-btn mobile-close" onClick={() => setMobileNav(false)}><X size={18}/></button>
        </div>
        <div className="workspace-switch"><div className="workspace-avatar">DI</div><div className="workspace-copy"><b>Delivery research</b><span>Personal workspace</span></div><ChevronDown size={15}/></div>
        <nav className="nav-groups">
          {nav.map(group => <div className="nav-group" key={group.label}>
            <div className="nav-label">{group.label}</div>
            {group.items.map(item => {
              const Icon = item.icon;
              return <button key={item.id} className={`nav-item ${page === item.id ? "active" : ""}`} onClick={() => {setPage(item.id);setMobileNav(false);}}>
                <Icon size={17}/><span>{item.name}</span>{item.id === "data" && audit && <span className="nav-dot"/>}
              </button>;
            })}
          </div>)}
        </nav>
        <div className="sidebar-bottom">
          <div className="pipeline-card"><div className="pipeline-top"><span className="pulse-dot"/> Pipeline status</div><strong>{audit ? "Dataset connected" : "Awaiting dataset"}</strong><p>{audit ? `${audit.rows.toLocaleString()} rows audited` : "Upload your CSV to begin analysis."}</p><button onClick={() => fileRef.current?.click()}>{audit ? "Replace dataset" : "Upload dataset"} <ArrowUpRight size={14}/></button></div>
          <div className="user-row"><div className="user-avatar">PC</div><div><b>Research team</b><span>Project workspace</span></div><Settings2 size={16}/></div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left"><button className="icon-btn mobile-menu" onClick={() => setMobileNav(true)}><Menu size={19}/></button><div className="breadcrumb">Workspace <span>/</span> <b>{activeTitle}</b></div></div>
          <div className="topbar-right"><div className="system-status"><span className="pulse-dot"/> API connected when available</div><button className="icon-btn theme-toggle" aria-label={`Switch to ${theme==="dark"?"light":"dark"} mode`} title={`Switch to ${theme==="dark"?"light":"dark"} mode`} onClick={()=>setTheme(theme==="dark"?"light":"dark")}>{theme==="dark"?<Sun size={17}/>:<Moon size={17}/>}</button><button className="icon-btn" title="Notifications"><Bell size={18}/></button><div className="top-avatar">PC</div></div>
        </header>

        <div className="page-wrap">
          {error && <div className="error-banner"><AlertCircle size={17}/>{error}<button onClick={() => setError("")}><X size={15}/></button></div>}
          {page === "overview" && <Overview audit={audit} analytics={analytics} model={model} onUpload={() => fileRef.current?.click()} onNavigate={setPage}/>}
          {page === "data" && <DataPage audit={audit} filteredColumns={filteredColumns} search={search} setSearch={setSearch} tracking={trackingAnalytics} onUpload={() => fileRef.current?.click()}/>}
          {page === "analytics" && <AnalyticsPage audit={audit} analytics={analytics} tracking={trackingAnalytics} trackingError={trackingAnalyticsError} onUpload={() => fileRef.current?.click()} onFocusColumn={name=>{setSearch(name);setPage("data");}}/>}
          {page === "predict" && <PredictPage audit={audit} model={model} prediction={prediction} training={training} onTrain={handleTrain} onPredict={handlePredict} onUpload={() => fileRef.current?.click()}/>}
          {page === "models" && <ModelPerformancePage model={model} onNavigate={setPage}/>}
          {page === "tracking" && <TrackingPage audit={audit} onUpload={() => fileRef.current?.click()} onError={setError}/>}
          {page === "explain" && <ExplainabilityPage model={model} explanation={explanation} onExplain={handleExplain} onNavigate={setPage}/>}
          {page === "robustness" && <RobustnessPage model={model} onNavigate={setPage}/>}
          <footer className="footer"><span>DELIVERY INTELLIGENCE <i>•</i> RESEARCH BUILD</span><span>Review selected features for leakage <ShieldCheck size={13}/></span></footer>
        </div>
      </main>
      <input ref={fileRef} type="file" accept=".csv,text/csv" hidden onChange={e => handleFile(e.target.files?.[0])}/>
      {(uploading||restoring) && <div className="upload-overlay"><div className="upload-modal"><div className="spinner"/><b>{uploading?"Auditing your dataset":"Restoring saved dataset"}</b><span>{uploading?"Reading columns, missing values and data types…":"Loading the latest local CSV; large files can take a minute."}</span></div></div>}
    </div>
  );
}

function PageHeading({eyebrow,title,subtitle,action}:{eyebrow:string;title:string;subtitle:string;action?:React.ReactNode}) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>{action && <div className="heading-action">{action}</div>}</div>;
}
function Metric({label,value,detail,icon,accent="blue"}:{label:string;value:string;detail:string;icon:React.ReactNode;accent?:string}) {
  return <div className="metric-card"><div className={`metric-icon ${accent}`}>{icon}</div><div className="metric-label">{label}</div><div className="metric-value">{value}</div><div className="metric-detail">{detail}</div></div>;
}
function Overview({audit,analytics,model,onUpload,onNavigate}:{audit:Audit|null;analytics:Analytics|null;model:ModelResult|null;onUpload:()=>void;onNavigate:(p:string)=>void}) {
  return <>
    <PageHeading eyebrow="RESEARCH OVERVIEW" title="Delivery intelligence" subtitle="A transparent workspace for understanding and predicting e-commerce delivery delays." action={<button className="primary-btn" onClick={onUpload}><UploadCloud size={16}/> Import dataset</button>}/>
    {!audit ? <div className="welcome-card"><div className="welcome-copy"><div className="welcome-chip"><Sparkles size={13}/> PROJECT WORKSPACE</div><h2>Turn delivery data into<br/><em>actionable intelligence.</em></h2><p>Start by importing your e-commerce CSV. The audit will profile its structure, data quality, date coverage and numeric fields before any modelling begins.</p><button className="primary-btn large" onClick={onUpload}><UploadCloud size={17}/> Upload your dataset <ArrowUpRight size={16}/></button><div className="welcome-foot"><ShieldCheck size={14}/> No fabricated metrics. Every result is computed from your data.</div></div><div className="welcome-visual"><div className="orbit orbit-one"/><div className="orbit orbit-two"/><div className="visual-core"><PackageCheck size={35}/></div><div className="float-node node-a"><Activity size={17}/><span>Data signals</span></div><div className="float-node node-b"><BrainCircuit size={17}/><span>ML intelligence</span></div><div className="float-node node-c"><BarChart3 size={17}/><span>Insights</span></div></div></div>
    : <>
      <div className="dataset-banner"><div className="dataset-symbol"><FileCheck2 size={20}/></div><div className="dataset-copy"><b>{audit.filename}</b><span>Dataset audited successfully · {audit.memory_mb} MB in memory</span></div><span className="status-pill success"><span/> AUDITED</span><button className="text-btn" onClick={() => onNavigate("data")}>View audit <ArrowUpRight size={14}/></button></div>
      <div className="metric-grid">
        <Metric label="Total records" value={audit.rows.toLocaleString()} detail="Rows in uploaded dataset" icon={<Database size={18}/>} />
        <Metric label="Data attributes" value={String(audit.columns)} detail="Detected columns" icon={<Boxes size={18}/>} accent="violet"/>
        <Metric label="Missing values" value={`${audit.missing_pct}%`} detail={`${audit.total_missing.toLocaleString()} empty cells`} icon={<AlertCircle size={18}/>} accent="amber"/>
        <Metric label="Duplicate rows" value={audit.duplicate_pct+"%"} detail={`${audit.duplicate_rows.toLocaleString()} exact duplicates`} icon={<FileSpreadsheet size={18}/>} accent="teal"/>
      </div>
      <div className="content-grid">
        <section className="panel chart-panel"><div className="panel-head"><div><h3>Order volume over time</h3><p>Monthly records from the first detected date column</p></div><span className="panel-tag">DESCRIPTIVE</span></div>
          {analytics?.time_series?.length ? <div className="chart-area"><ResponsiveContainer width="100%" height="100%"><AreaChart data={analytics.time_series}><defs><linearGradient id="ordersFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#5b8def" stopOpacity={0.28}/><stop offset="100%" stopColor="#5b8def" stopOpacity={0}/></linearGradient></defs><CartesianGrid stroke="#202d40" vertical={false}/><XAxis dataKey="period" tick={{fill:"#7e8da4",fontSize:11}} axisLine={false} tickLine={false} minTickGap={28}/><YAxis tick={{fill:"#7e8da4",fontSize:11}} axisLine={false} tickLine={false} width={42}/><Tooltip contentStyle={{background:"#101b2b",border:"1px solid #26364c",borderRadius:10,color:"#e8eef8"}}/><Area type="monotone" dataKey="orders" stroke="#6b9cff" strokeWidth={2.5} fill="url(#ordersFill)"/></AreaChart></ResponsiveContainer></div> : <div className="empty-chart"><Clock3 size={22}/><b>No reliable date series detected</b><span>Check the date candidates in Data quality.</span></div>}
          <div className="chart-note"><span className="legend-dot"/> Monthly row count <span className="note-muted">{analytics?.date_column_used ? `· ${analytics.date_column_used}` : ""}</span></div>
        </section>
        <section className="panel readiness-panel"><div className="panel-head"><div><h3>Research readiness</h3><p>What is available at this stage</p></div><span className="panel-tag">LIVE STATUS</span></div>
          <div className="readiness-list"><Readiness label="Dataset ingestion" state="complete" detail="CSV parsed and profiled"/><Readiness label="Data quality audit" state="complete" detail="Missingness and duplicates measured"/><Readiness label="Target mapping" state={model?"complete":"pending"} detail={model?`Selected target: ${model.target_column} · verify label meaning`:"Choose and verify a delivery outcome"}/><Readiness label="Feature review" state="pending" detail={model?`${model.features.length} inputs selected · review for leakage`:"Select checkout-time inputs"}/><Readiness label="Model evaluation" state={model?"complete":"locked"} detail={model?`${model.model_name} · ${model.test_rows.toLocaleString()} holdout rows`:"Train a classifier to calculate results"}/></div>
          <button className="secondary-btn full" onClick={() => onNavigate(model?"models":"predict")}>{model?"Review model results":"Configure model"} <ArrowUpRight size={14}/></button>
        </section>
      </div>
      <div className="notice"><AlertCircle size={16}/><div><b>Interpretation note</b><span>Order volume is descriptive, not a delivery-performance metric. Delay rates and model scores will appear only after outcome labels and feature timing are verified.</span></div></div>
    </>}
  </>;
}
function Readiness({label,state,detail}:{label:string;state:string;detail:string}) {
  return <div className="readiness-row"><div className={`readiness-check ${state}`}>{state==="complete"?<Check size={13}/>:state==="locked"?<X size={12}/>:<Clock3 size={12}/>}</div><div className="readiness-copy"><b>{label}</b><span>{detail}</span></div><span className={`readiness-state ${state}`}>{state==="complete"?"DONE":state==="locked"?"LOCKED":"PENDING"}</span></div>;
}
function DataPage({audit,filteredColumns,search,setSearch,tracking,onUpload}:{audit:Audit|null;filteredColumns:Audit["columns_profile"];search:string;setSearch:(s:string)=>void;tracking:TrackingAnalytics|null;onUpload:()=>void}) {
  if (!audit) return <EmptyState title="Start with your dataset" text="Upload a CSV to inspect its columns, missing values, duplicates and date coverage." onUpload={onUpload}/>;
  const missingness=[...audit.columns_profile].sort((a,b)=>b.missing_pct-a.missing_pct).slice(0,12);
  return <>
    <PageHeading eyebrow="DATA FOUNDATION" title="Data quality" subtitle="Inspect the uploaded source before defining targets or training any model." action={<button className="primary-btn" onClick={onUpload}><UploadCloud size={16}/> Replace CSV</button>}/>
    <div className="metric-grid compact"><Metric label="Rows" value={audit.rows.toLocaleString()} detail="Parsed records" icon={<Database size={17}/>}/><Metric label="Columns" value={String(audit.columns)} detail="Detected fields" icon={<Boxes size={17}/>} accent="violet"/><Metric label="Missing cells" value={audit.total_missing.toLocaleString()} detail={`${audit.missing_pct}% of all cells`} icon={<AlertCircle size={17}/>} accent="amber"/><Metric label="Exact duplicates" value={audit.duplicate_rows.toLocaleString()} detail={`${audit.duplicate_pct}% of rows`} icon={<FileSpreadsheet size={17}/>} accent="teal"/></div>
    <section className="panel section-panel"><div className="panel-head"><div><h3>Missingness by field</h3><p>Click a bar to focus the column profile below.</p></div><span className="panel-tag">INTERACTIVE AUDIT</span></div><div className="chart-area tall"><ResponsiveContainer width="100%" height="100%"><BarChart data={missingness} layout="vertical" margin={{left:14,right:20}}><CartesianGrid stroke="#202d40" horizontal={false}/><XAxis type="number" domain={[0,100]} tickFormatter={value=>`${value}%`} tick={{fill:"#7e8da4",fontSize:10}} axisLine={false} tickLine={false}/><YAxis type="category" dataKey="name" width={150} tick={{fill:"#9aa9bd",fontSize:9}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#101b2b",border:"1px solid #26364c",borderRadius:10}} formatter={value=>[`${Number(value).toFixed(1)}%`,"Missing"]}/><Bar dataKey="missing_pct" name="Missing share" fill="#e1a95e" radius={[0,4,4,0]} onClick={item=>setSearch(String(item?.name||""))}/></BarChart></ResponsiveContainer></div></section>
    {tracking&&(tracking.status_completion_conflicts>0||tracking.delivery_dates_on_open>0)&&<div className="notice"><AlertCircle size={16}/><div><b>Cross-field consistency checks</b><span>{tracking.status_completion_conflicts.toLocaleString()} rows have conflicting delivery-status and completion fields; {tracking.delivery_dates_on_open.toLocaleString()} incomplete rows contain a delivery date. Tracking outcome statistics follow the completion flag and ignore dates on incomplete rows.</span></div></div>}
    <section className="panel section-panel"><div className="panel-head"><div><h3>Column profile</h3><p>Search and inspect inferred data types and completeness</p></div><span className="panel-tag">{audit.columns} FIELDS</span></div><div className="table-toolbar"><div className="search-box"><Search size={15}/><input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search columns…"/></div><span>{filteredColumns.length} of {audit.columns} columns</span></div><div className="table-wrap"><table><thead><tr><th>Column</th><th>Data type</th><th>Unique</th><th>Missing</th><th>Completeness</th><th>Examples</th></tr></thead><tbody>{filteredColumns.map(c=><tr key={c.name}><td className="field-name">{c.name}</td><td><span className="type-pill">{c.dtype}</span></td><td>{c.unique.toLocaleString()}</td><td>{c.missing.toLocaleString()} <span className="muted">({c.missing_pct}%)</span></td><td><div className="progress-cell"><div className="progress-track"><div className={`progress-fill ${c.missing_pct>30?"warn":""}`} style={{width:`${100-c.missing_pct}%`}}/></div><span>{100-c.missing_pct}%</span></div></td><td className="examples">{c.examples.map(x=>String(x)).join(", ") || "—"}</td></tr>)}</tbody></table></div></section>
    <div className="content-grid lower-grid"><section className="panel section-panel"><div className="panel-head"><div><h3>Date candidates</h3><p>Columns with at least 50% parseable date values</p></div></div>{audit.date_candidates.length ? audit.date_candidates.map(d=><div className="date-candidate" key={d.column}><div><b>{d.column}</b><span>{d.min || "—"} → {d.max || "—"}</span></div><span className="status-pill success">{d.valid_pct}% valid</span></div>) : <div className="muted-block">No date candidates detected automatically.</div>}</section>
      <section className="panel section-panel"><div className="panel-head"><div><h3>Audit notes</h3><p>Checks to complete before modelling</p></div></div>{audit.warnings.map((w,i)=><div className="audit-note" key={i}><AlertCircle size={15}/><span>{w}</span></div>)}</section></div>
    <section className="panel section-panel preview-panel"><div className="panel-head"><div><h3>Data preview</h3><p>First 8 parsed rows · values truncated for readability</p></div><span className="panel-tag">SAMPLE</span></div><div className="table-wrap"><table><thead><tr>{audit.columns_profile.slice(0,8).map(c=><th key={c.name}>{c.name}</th>)}</tr></thead><tbody>{audit.preview.map((r,i)=><tr key={i}>{audit.columns_profile.slice(0,8).map(c=><td key={c.name}>{String(r[c.name] ?? "—").slice(0,48)}</td>)}</tr>)}</tbody></table></div></section>
  </>;
}
function AnalyticsPage({audit,analytics,tracking,trackingError,onUpload,onFocusColumn}:{audit:Audit|null;analytics:Analytics|null;tracking:TrackingAnalytics|null;trackingError:string;onUpload:()=>void;onFocusColumn:(name:string)=>void}) {
  const [range,setRange] = useState("all");
  const [metric,setMetric] = useState<"orders"|"delayed"|"delay_rate">("orders");
  const [chartType,setChartType] = useState<"area"|"bar">("area");
  const [selectedMonth,setSelectedMonth] = useState("");
  const [selectedStatus,setSelectedStatus] = useState("");
  const [statusChartType,setStatusChartType] = useState<"bar"|"area">("bar");
  const sourceSeries = tracking?.monthly.length
    ? tracking.monthly.map(item=>({period:item.month,orders:item.orders,delayed:item.delayed,delay_rate:item.delay_rate,completed:item.completed}))
    : (analytics?.time_series||[]).map(item=>({period:item.period,orders:item.orders,delayed:null,delay_rate:null,completed:null}));
  const series = range==="all"?sourceSeries:sourceSeries.slice(-Number(range));
  const chartData = series.map(item=>({...item,value:metric==="orders"?item.orders:metric==="delayed"?(item.delayed??0):(item.delay_rate===null?null:item.delay_rate*100)}));
  const profile = audit?.columns_profile || [];
  const missingness = [...profile].sort((a,b)=>b.missing_pct-a.missing_pct).slice(0,12);
  const activeMonth = tracking?.monthly.find(item=>item.month===selectedMonth);
  if(!audit) return <EmptyState title="Analytics need a dataset" text="Import your source CSV to calculate descriptive trends from actual records." onUpload={onUpload}/>;
  return <>
    <PageHeading eyebrow="EXPLORATORY ANALYSIS" title="Analytics" subtitle="Explore delivery volumes, recorded delay outcomes, and data quality in the uploaded CSV."/>
    <div className="notice"><Activity size={16}/><div><b>Descriptive analysis · not a live forecast</b><span>{tracking?.summary || trackingError || analytics?.message || "Upload a dataset with delivery outcome fields to compare delays."}</span></div></div>
    <section className="panel section-panel">
      <div className="panel-head"><div><h3>Monthly delivery trends</h3><p>{tracking?"Delay rates use completed orders only. Select the measure and period to explore.":analytics?.date_column_used?`Monthly rows using ${analytics.date_column_used}`:"No usable date field identified."}</p></div><span className="panel-tag">INTERACTIVE</span></div>
      <div className="chart-controls">
        <label>Measure<select value={metric} onChange={event=>setMetric(event.target.value as "orders"|"delayed"|"delay_rate")}><option value="orders">Orders</option><option value="delayed" disabled={!tracking}>Delayed completed</option><option value="delay_rate" disabled={!tracking}>Delay rate</option></select></label>
        <label>Time window<select value={range} onChange={event=>setRange(event.target.value)}><option value="all">All available</option><option value="12">Latest 12 months</option><option value="6">Latest 6 months</option><option value="3">Latest 3 months</option></select></label>
        <div className="chart-toggle" role="group" aria-label="Chart type"><button className={chartType==="area"?"selected":""} onClick={()=>setChartType("area")}>Area</button><button className={chartType==="bar"?"selected":""} onClick={()=>setChartType("bar")}>Bars</button></div>
      </div>
      {chartData.length ? <div className="chart-area tall"><ResponsiveContainer width="100%" height="100%">
        {chartType==="area"
          ? <AreaChart data={chartData}><CartesianGrid stroke="#202d40" vertical={false}/><XAxis dataKey="period" tick={{fill:"#7e8da4",fontSize:10}} axisLine={false} tickLine={false} minTickGap={24}/><YAxis tick={{fill:"#7e8da4",fontSize:10}} axisLine={false} tickLine={false} tickFormatter={value=>metric==="delay_rate"?`${value}%`:Number(value).toLocaleString()}/><Tooltip contentStyle={{background:"#101b2b",border:"1px solid #26364c",borderRadius:10}} formatter={value=>[metric==="delay_rate"?`${Number(value).toFixed(1)}%`:Number(value).toLocaleString(),metric==="orders"?"Orders":metric==="delayed"?"Delayed completed":"Delay rate"]}/><Area type="monotone" dataKey="value" stroke={metric==="delay_rate"?"#f3bd70":"#6b9cff"} fill={metric==="delay_rate"?"#edaa4b":"#477de5"} fillOpacity={0.18} connectNulls/></AreaChart>
          : <BarChart data={chartData}><CartesianGrid stroke="#202d40" vertical={false}/><XAxis dataKey="period" tick={{fill:"#7e8da4",fontSize:10}} axisLine={false} tickLine={false} minTickGap={24}/><YAxis tick={{fill:"#7e8da4",fontSize:10}} axisLine={false} tickLine={false} tickFormatter={value=>metric==="delay_rate"?`${value}%`:Number(value).toLocaleString()}/><Tooltip contentStyle={{background:"#101b2b",border:"1px solid #26364c",borderRadius:10}} formatter={value=>[metric==="delay_rate"?`${Number(value).toFixed(1)}%`:Number(value).toLocaleString(),metric==="orders"?"Orders":metric==="delayed"?"Delayed completed":"Delay rate"]}/><Bar dataKey="value" fill={metric==="delay_rate"?"#edaa4b":"#6b9cff"} radius={[4,4,0,0]} onClick={item=>setSelectedMonth(String(item?.period||""))}/></BarChart>}
      </ResponsiveContainer></div>:<div className="empty-chart">No time series available. Review the date candidates in Data quality.</div>}
      {activeMonth&&<div className="selection-summary"><b>{activeMonth.month}</b><span>{activeMonth.orders.toLocaleString()} orders</span><span>{activeMonth.completed.toLocaleString()} completed</span><span>{activeMonth.delayed.toLocaleString()} delayed</span><span>{activeMonth.delay_rate===null?"Delay rate unavailable":`${(activeMonth.delay_rate*100).toFixed(1)}% delay rate`}</span><button onClick={()=>setSelectedMonth("")}>Clear</button></div>}
    </section>
    <section className="panel section-panel">
      <div className="panel-head"><div><h3>Recorded status mix</h3><p>Counts from the CSV; click a bar to inspect a status.</p></div><div className="chart-toggle" role="group" aria-label="Status chart type"><button className={statusChartType==="bar"?"selected":""} onClick={()=>setStatusChartType("bar")}>Bars</button><button className={statusChartType==="area"?"selected":""} onClick={()=>setStatusChartType("area")}>Area</button></div></div>
      {tracking?.statuses.length ? <div className="chart-area tall"><ResponsiveContainer width="100%" height="100%">
        {statusChartType==="bar"
          ? <BarChart data={tracking.statuses} margin={{top:8,right:10,bottom:28,left:0}}><CartesianGrid stroke="#202d40" vertical={false}/><XAxis dataKey="status" interval={0} angle={-22} textAnchor="end" height={62} tick={{fill:"#7e8da4",fontSize:9}} axisLine={false} tickLine={false}/><YAxis tick={{fill:"#7e8da4",fontSize:10}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#101b2b",border:"1px solid #26364c",borderRadius:10}}/><Bar dataKey="count" name="Orders" fill="#6b9cff" radius={[4,4,0,0]} onClick={item=>setSelectedStatus(String(item?.status||""))}><Cell fill="#6b9cff"/></Bar></BarChart>
          : <AreaChart data={tracking.statuses}><CartesianGrid stroke="#202d40" vertical={false}/><XAxis dataKey="status" interval={0} angle={-22} textAnchor="end" height={62} tick={{fill:"#7e8da4",fontSize:9}} axisLine={false} tickLine={false}/><YAxis tick={{fill:"#7e8da4",fontSize:10}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#101b2b",border:"1px solid #26364c",borderRadius:10}}/><Area type="monotone" dataKey="count" name="Orders" stroke="#6b9cff" fill="#477de5" fillOpacity={0.18}/></AreaChart>}
      </ResponsiveContainer></div>:<div className="empty-chart">{trackingError||"No delivery-status field was found for this dataset."}</div>}
      {selectedStatus&&<div className="selection-summary"><b>{selectedStatus}</b><span>{tracking?.statuses.find(item=>item.status===selectedStatus)?.count.toLocaleString()} rows</span><button onClick={()=>setSelectedStatus("")}>Clear</button></div>}
    </section>
    <section className="panel section-panel">
      <div className="panel-head"><div><h3>Missingness by field</h3><p>Highest missing-value shares. Select a bar to filter the column profile.</p></div><span className="panel-tag">DATA QUALITY</span></div>
      <div className="chart-area tall"><ResponsiveContainer width="100%" height="100%"><BarChart data={missingness} layout="vertical" margin={{left:14,right:20}}><CartesianGrid stroke="#202d40" horizontal={false}/><XAxis type="number" domain={[0,100]} tickFormatter={value=>`${value}%`} tick={{fill:"#7e8da4",fontSize:10}} axisLine={false} tickLine={false}/><YAxis type="category" dataKey="name" width={150} tick={{fill:"#9aa9bd",fontSize:9}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#101b2b",border:"1px solid #26364c",borderRadius:10}} formatter={value=>[`${Number(value).toFixed(1)}%`,"Missing"]}/><Bar dataKey="missing_pct" name="Missing share" fill="#e1a95e" radius={[0,4,4,0]} onClick={item=>onFocusColumn(String(item?.name||""))}/></BarChart></ResponsiveContainer></div>
    </section>
    <section className="panel section-panel"><div className="panel-head"><div><h3>Numeric field summary</h3><p>First 20 numeric fields · computed directly from uploaded data</p></div></div><div className="table-wrap"><table><thead><tr><th>Field</th><th>Mean</th><th>Median</th><th>Minimum</th><th>Maximum</th></tr></thead><tbody>{audit.numeric_summary.map(n=><tr key={n.name}><td className="field-name">{n.name}</td><td>{n.mean ?? "—"}</td><td>{n.median ?? "—"}</td><td>{n.min ?? "—"}</td><td>{n.max ?? "—"}</td></tr>)}</tbody></table></div></section>
  </>;
}
async function requestJson<T>(url:string):Promise<T> {
  const response = await fetch(url);
  const data = await response.json();
  if(!response.ok) throw new Error(data.detail || "The tracking request failed.");
  return data as T;
}
function TrackingPage({audit,onUpload,onError}:{audit:Audit|null;onUpload:()=>void;onError:(message:string)=>void}) {
  const [analytics,setAnalytics] = useState<TrackingAnalytics|null>(null);
  const [orders,setOrders] = useState<OrderPage|null>(null);
  const [detail,setDetail] = useState<OrderDetail|null>(null);
  const [searchInput,setSearchInput] = useState("");
  const [search,setSearch] = useState("");
  const [status,setStatus] = useState("");
  const [page,setPage] = useState(1);
  const [loading,setLoading] = useState(false);
  const [ordersLoading,setOrdersLoading] = useState(true);
  const pageSize = 25;

  useEffect(()=>{
    if(!audit) return;
    let active=true;
    requestJson<TrackingAnalytics>("/api/tracking/analytics")
      .then(data=>{if(active)setAnalytics(data);})
      .catch(error=>{if(active)onError(error instanceof Error?error.message:"Could not load delivery statistics.");});
    return ()=>{active=false;};
  },[audit,onError]);

  useEffect(()=>{
    if(!audit) return;
    let active=true;
    setOrdersLoading(true);
    const params=new URLSearchParams({page:String(page),page_size:String(pageSize),search,status});
    requestJson<OrderPage>(`/api/orders?${params.toString()}`)
      .then(data=>{if(active)setOrders(data);})
      .catch(error=>{if(active)onError(error instanceof Error?error.message:"Could not load package records.");})
      .finally(()=>{if(active)setOrdersLoading(false);});
    return ()=>{active=false;};
  },[audit,page,pageSize,search,status,onError]);

  async function selectOrder(orderId:string) {
    setLoading(true);
    setDetail(null);
    try {
      const params=new URLSearchParams({order_id:orderId});
      setDetail(await requestJson<OrderDetail>(`/api/orders/detail?${params.toString()}`));
    } catch(error) {
      onError(error instanceof Error?error.message:"Could not load this package.");
    } finally {
      setLoading(false);
    }
  }

  if(!audit) return <EmptyState title="Package tracking needs a CSV" text="Upload your order export to browse package IDs, statuses and delivery outcomes." onUpload={onUpload}/>;
  const statusChart=analytics?.statuses || [];
  const totalPages=Math.max(1,Math.ceil((orders?.total||0)/pageSize));
  const detailSummary=detail?.completed
    ? `The CSV snapshot marks this order as completed${detail.actual_delivery_date?` on ${detail.actual_delivery_date}`:""}${detail.is_delayed===true?" and delayed":detail.is_delayed===false?" and on time":"; no delay outcome is available"}.`
    : detail?.estimated_arrival_date
      ? `The CSV records “${detail.status}”${detail.delay_flag_recorded===true?"; its delay flag is set, although it is not marked complete":""}. Comparable completed deliveries suggest an estimated arrival of ${detail.estimated_arrival_date}${detail.estimated_remaining_days!==null?(detail.estimated_remaining_days===0?" (estimated for today)":` (about ${detail.estimated_remaining_days} more days)`):detail.eta_overdue_days!==null?` (the historical ETA date is ${detail.eta_overdue_days} days before today)`: ""}. The estimate uses ${detail.cohort_count.toLocaleString()} completed deliveries${detail.probability_by_promised_date!==null?`; ${Math.round(detail.probability_by_promised_date*100)}% arrived by the promised transit day`:""}. This CSV snapshot has no current carrier scan; if the date has passed, it cannot establish a new remaining-time estimate.`
      : `The CSV records “${detail?.status}”. There is not enough usable order-date or delivery-history data to estimate an arrival date.`;

  return <>
    <PageHeading eyebrow="CSV SNAPSHOT · NOT LIVE" title="Package tracking" subtitle="Search and open any package in the uploaded CSV. Statuses are recorded values; estimated arrival windows use historical completed orders."/>
    <div className="notice"><MapPin size={16}/><div><b>CSV records only</b><span>{analytics?.source||"Tracking is based on the uploaded file and does not receive live carrier scans."}{analytics?.order_date_range.start&&analytics.order_date_range.end?` Order dates in this file: ${analytics.order_date_range.start} to ${analytics.order_date_range.end}.`:""} Missing ETA details are estimated where possible; source rows are never overwritten.</span></div></div>
    {analytics&&<>
      <div className="metric-grid tracking-metrics">
        <Metric label="Packages in file" value={analytics.total_orders.toLocaleString()} detail="Rows with package records" icon={<PackageCheck size={17}/>}/>
        <Metric label="Completed" value={analytics.completed_orders.toLocaleString()} detail={`${analytics.open_orders.toLocaleString()} not marked complete`} icon={<Check size={17}/>} accent="teal"/>
        <Metric label="Delayed deliveries" value={analytics.delayed_completed.toLocaleString()} detail={analytics.delay_rate===null?"No completed outcomes":`${(analytics.delay_rate*100).toFixed(1)}% of completed orders`} icon={<Clock3 size={17}/>} accent="amber"/>
        <Metric label="Average late days" value={analytics.average_late_days===null?"—":`${analytics.average_late_days} days`} detail="Delayed completed orders only" icon={<Activity size={17}/>} accent="violet"/>
      </div>
      <p className="tracking-summary">{analytics.summary}</p>
      <div className="content-grid tracking-charts">
        <section className="panel chart-panel"><div className="panel-head"><div><h3>Orders and delay rate over time</h3><p>Monthly rows; delay rate uses completed orders only</p></div><span className="panel-tag">CSV DATA</span></div>
          {analytics.monthly.length?<div className="chart-area"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={analytics.monthly}><CartesianGrid stroke="#202d40" vertical={false}/><XAxis dataKey="month" tick={{fill:"#7e8da4",fontSize:10}} axisLine={false} tickLine={false} minTickGap={30}/><YAxis yAxisId="orders" tick={{fill:"#7e8da4",fontSize:10}} axisLine={false} tickLine={false}/><YAxis yAxisId="rate" orientation="right" domain={[0,1]} tickFormatter={value=>`${Math.round(value*100)}%`} tick={{fill:"#7e8da4",fontSize:10}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#101b2b",border:"1px solid #26364c",borderRadius:10}} formatter={(value,name)=>[name==="delay_rate"?`${(Number(value)*100).toFixed(1)}%`:Number(value).toLocaleString(),name==="delay_rate"?"Delay rate":"Orders"]}/><Legend/><Area yAxisId="orders" type="monotone" dataKey="orders" name="Orders" stroke="#6b9cff" fill="#477de5" fillOpacity={0.14}/><Line yAxisId="rate" type="monotone" dataKey="delay_rate" name="Delay rate" stroke="#f3bd70" connectNulls dot={false}/></ComposedChart></ResponsiveContainer></div>:<div className="empty-chart">No usable order-date column was found.</div>}
        </section>
        <section className="panel chart-panel"><div className="panel-head"><div><h3>Packages by recorded status</h3><p>Most frequent statuses in this CSV</p></div><span className="panel-tag">SNAPSHOT</span></div>
          {statusChart.length?<div className="chart-area"><ResponsiveContainer width="100%" height="100%"><BarChart data={statusChart} margin={{top:8,right:8,bottom:32,left:0}}><CartesianGrid stroke="#202d40" vertical={false}/><XAxis dataKey="status" interval={0} angle={-24} textAnchor="end" height={65} tick={{fill:"#7e8da4",fontSize:9}} axisLine={false} tickLine={false}/><YAxis tick={{fill:"#7e8da4",fontSize:10}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#101b2b",border:"1px solid #26364c",borderRadius:10}}/><Bar dataKey="count" name="Packages" fill="#6b9cff" radius={[4,4,0,0]}><Cell fill="#6b9cff"/></Bar></BarChart></ResponsiveContainer></div>:<div className="empty-chart">No status field was found in the CSV.</div>}
        </section>
      </div>
    </>}
    <section className="panel section-panel package-list">
      <div className="panel-head"><div><h3>Packages</h3><p>Click an order ID to view its CSV status, estimated arrival probability and recorded signals.</p></div><span className="panel-tag">{ordersLoading?"LOADING…":`${(orders?.total||0).toLocaleString()} MATCHES`}</span></div>
      <form className="package-filters" onSubmit={event=>{event.preventDefault();setPage(1);setSearch(searchInput.trim());}}>
        <div className="search-box"><Search size={15}/><input value={searchInput} onChange={event=>setSearchInput(event.target.value)} placeholder="Find order or tracking ID"/></div>
        <select aria-label="Filter by package status" value={status} onChange={event=>{setPage(1);setStatus(event.target.value);}}><option value="">All statuses</option>{(orders?.statuses||[]).map(value=><option key={value} value={value}>{value}</option>)}</select>
        <button className="secondary-btn" type="submit">Search</button>
      </form>
      <div className="table-wrap"><table><thead><tr><th>Order / package ID</th><th>Order date</th><th>Package</th><th>Destination</th><th>Recorded status</th><th>Outcome</th></tr></thead><tbody>{orders?.items.map(order=><tr key={order.order_id}><td><button className="package-link" onClick={()=>void selectOrder(order.order_id)}>{order.order_id}</button></td><td>{order.order_date||"—"}</td><td>{order.product||"—"}</td><td>{order.destination||"—"}</td><td><span className="type-pill">{order.status}</span></td><td>{order.completed?(order.delayed===true?<span className="outcome late">Delayed</span>:order.delayed===false?<span className="outcome on-time">On time</span>:"Completed"):"No completed outcome"}</td></tr>)}</tbody></table>{!orders?.items.length&&<div className="empty-chart package-empty">{orders?"No packages match this search.":"Loading package records…"}</div>}</div>
      <div className="package-pagination"><span>{ordersLoading?"Loading package records…":`Page ${page} of ${totalPages} · ${orders?.total.toLocaleString()||"0"} packages`}</span><div><button className="secondary-btn" disabled={ordersLoading||page<=1} onClick={()=>setPage(value=>Math.max(1,value-1))}>Previous</button><button className="secondary-btn" disabled={ordersLoading||page>=totalPages} onClick={()=>setPage(value=>Math.min(totalPages,value+1))}>Next</button></div></div>
    </section>
    {loading&&<div className="panel section-panel tracking-loading">Loading package details…</div>}
    {detail&&<section className="panel section-panel tracking-detail">
      <div className="panel-head"><div><div className="eyebrow">PACKAGE RECORD</div><h3>{detail.order_id}</h3><p>{detail.product||detail.category||"Order"}{detail.destination?` · ${detail.destination}`:""}</p></div><button className="secondary-btn" onClick={()=>setDetail(null)}>Close</button></div>
      <div className="notice tracking-summary-box"><Activity size={16}/><div><b>What the CSV says</b><span>{detailSummary}</span></div></div>
      <div className="metric-grid compact tracking-detail-metrics">
        <Metric label="Recorded status" value={detail.status} detail={detail.completed?"Marked complete in CSV":"Not marked complete in CSV"} icon={<PackageCheck size={17}/>}/>
        <Metric label={detail.completed?"Actual delivery":"Estimated arrival"} value={detail.completed?(detail.actual_delivery_date||"Date not recorded"):(detail.estimated_arrival_date||"Not available")} detail={detail.completed?"Recorded completed order date":detail.estimated_remaining_days!==null?(detail.estimated_remaining_days===0?"Historical estimate is for today":`Historical estimate: about ${detail.estimated_remaining_days} more days`):detail.eta_overdue_days!==null?`Historical ETA date is ${detail.eta_overdue_days} days before today`:"Based on comparable historical transit time"} icon={<Clock3 size={17}/>} accent="violet"/>
        <Metric label="Chance by promised date" value={detail.completed?"—":detail.probability_by_promised_date===null?"Not available":`${(detail.probability_by_promised_date*100).toFixed(1)}%`} detail={detail.completed?"This package has a recorded outcome":detail.probability_interval?`Approx. 95% interval ${(detail.probability_interval.lower*100).toFixed(1)}–${(detail.probability_interval.upper*100).toFixed(1)}% · n=${detail.cohort_count.toLocaleString()}`:`${detail.cohort_count.toLocaleString()} comparable completed orders`} icon={<Activity size={17}/>} accent="teal"/>
        <Metric label="Days vs promise" value={detail.completed&&detail.delay_days!==null?`${detail.delay_days>0?"+":""}${detail.delay_days} days`:"—"} detail={detail.completed?"Recorded delay days":"No confirmed delivery outcome"} icon={<Gauge size={17}/>} accent="amber"/>
      </div>
      {!detail.completed&&detail.arrival_probabilities.length>0&&<div className="panel tracking-subpanel"><div className="panel-head"><div><h3>Historical probability of arrival</h3><p>{detail.probability_label} Reference group: {detail.cohort_description}.</p></div></div><div className="chart-area"><ResponsiveContainer width="100%" height="100%"><AreaChart data={detail.arrival_probabilities}><CartesianGrid stroke="#202d40" vertical={false}/><XAxis dataKey="transit_day" tickFormatter={value=>`Day ${value}`} tick={{fill:"#7e8da4",fontSize:10}} axisLine={false} tickLine={false}/><YAxis domain={[0,1]} tickFormatter={value=>`${Math.round(value*100)}%`} tick={{fill:"#7e8da4",fontSize:10}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#101b2b",border:"1px solid #26364c",borderRadius:10}} labelFormatter={value=>`Transit day ${value}`} formatter={value=>[`${(Number(value)*100).toFixed(1)}%`,"Historical cumulative share"]}/><Area type="monotone" dataKey="probability" stroke="#6b9cff" fill="#477de5" fillOpacity={0.18}/></AreaChart></ResponsiveContainer></div></div>}
      <div className="content-grid tracking-detail-grid">
        <section className="panel tracking-subpanel"><h3>{detail.is_delayed===true?"Delayed delivery: signals in the CSV":detail.delay_flag_recorded===true?"CSV delay flag: signals on record":detail.completed?"Operational signals on record":"Possible delay contributors"}</h3><p className="tracking-caveat">The CSV has no verified event-reason field (for example, a landslide report). These recorded flags and statistical signals may be relevant, but do not prove the cause.</p>{detail.possible_contributors.map(item=><div className="audit-note" key={item.factor}><AlertCircle size={15}/><span><b>{item.factor}:</b> {item.detail}</span></div>)}</section>
        <section className="panel tracking-subpanel"><h3>Recorded tracking timeline</h3><p className="tracking-caveat">Only dates present in the CSV are shown; scan events are not invented.</p><div className="timeline">{detail.timeline.map((item,index)=><div className={`timeline-item ${item.state}`} key={`${item.label}-${index}`}><span className="timeline-dot"/><div><b>{item.label}</b><span>{item.date||"No event timestamp in CSV"}</span></div></div>)}</div></section>
      </div>
      {detail.data_notes.map(note=><div className="audit-note" key={note}><AlertCircle size={15}/><span>{note}</span></div>)}
      <p className="result-footnote">{detail.source}</p>
    </section>}
  </>;
}
function PredictPage({audit,model,prediction,training,onTrain,onPredict,onUpload}:{audit:Audit|null;model:ModelResult|null;prediction:PredictionResult|null;training:boolean;onTrain:(target:string,features:string[],dateColumn:string)=>Promise<void>;onPredict:(values:Record<string,string>)=>Promise<void>;onUpload:()=>void}) {
  const [target,setTarget] = useState("");
  const [features,setFeatures] = useState<string[]>([]);
  const [dateColumn,setDateColumn] = useState("");
  const [values,setValues] = useState<Record<string,string>>({});

  useEffect(()=>{
    setValues(Object.fromEntries((model?.feature_schema || []).map(f=>[f.name,f.kind==="number"?"":(f.examples[0] || "")])));
  },[model]);

  if(!audit) return <EmptyState title="Training needs a dataset" text="Upload your labeled CSV before selecting a target and training a classifier." onUpload={onUpload}/>;
  const availableFeatures = audit.columns_profile.map(column=>column.name).filter(name=>name!==target && name!==dateColumn);
  const selectedFeatures = features.filter(name=>availableFeatures.includes(name));
  const toggleFeature = (name:string) => setFeatures(current=>current.includes(name)?current.filter(item=>item!==name):[...current,name]);
  return <>
    <PageHeading eyebrow="MODEL TRAINING & INFERENCE" title="Order prediction" subtitle="Choose the outcome and input fields, train a classifier, then generate an output from entered feature values."/>
    {!model && <div className="notice"><Activity size={16}/><div><b>Use verified inputs</b><span>Select the target and only fields that are available when the prediction would be made. The model reports the selected target labels; it does not assume which class means “late”.</span></div></div>}
    <div className="prediction-layout">
      <section className="panel section-panel training-panel">
        <div className="panel-head"><div><h3>Train a classification model</h3><p>Random forest · 80/20 holdout evaluation</p></div>{model&&<span className="status-pill success">TRAINED</span>}</div>
        <label className="field-control">Target column<select value={target} onChange={event=>{setTarget(event.target.value);setFeatures([]);if(event.target.value===dateColumn)setDateColumn("");}}><option value="">Select the outcome to predict</option>{audit.columns_profile.map(column=><option key={column.name} value={column.name}>{column.name}</option>)}</select></label>
        <label className="field-control">Optional chronological split<select value={dateColumn} onChange={event=>setDateColumn(event.target.value)}><option value="">Use stratified random split</option>{audit.date_candidates.filter(date=>date.column!==target).map(date=><option key={date.column} value={date.column}>{date.column}</option>)}</select><small>Choose only a verified order-time field. The latest 20% is held out for testing.</small></label>
        <div className="feature-picker"><div className="feature-picker-head"><b>Feature columns</b><span>{selectedFeatures.length} selected</span></div>{target?<div className="feature-options">{availableFeatures.map(name=><label key={name}><input type="checkbox" checked={selectedFeatures.includes(name)} onChange={()=>toggleFeature(name)}/><span>{name}</span></label>)}</div>:<p className="muted-block">Select a target first to choose input features.</p>}</div>
        <button className="primary-btn full" disabled={training||!target||selectedFeatures.length===0} onClick={()=>onTrain(target,selectedFeatures,dateColumn)}>{training?"Training model…":"Train model and show evaluation"}</button>
      </section>
      <aside className="panel section-panel safeguards-panel">
        <h3>Before training</h3>
        <div className="safeguard"><ShieldCheck/><div><b>Explicit feature mapping</b><span>Only selected columns are used; remove post-outcome fields to prevent leakage.</span></div></div>
        <div className="safeguard"><Clock3/><div><b>Holdout evaluation</b><span>{dateColumn?`Rows are split by time using ${dateColumn}.`:"A stratified random 20% test split is used unless you select a verified date column."}</span></div></div>
        <div className="safeguard"><AlertCircle/><div><b>Dataset-dependent results</b><span>Scores and predictions are calculated from the uploaded file and selected target.</span></div></div>
      </aside>
    </div>
    {model&&<>
      <section className="panel section-panel result-panel">
        <div className="panel-head"><div><h3>Holdout evaluation</h3><p>{model.model_name} · target: {model.target_column}</p></div><span className="panel-tag">{model.split_method}</span></div>
        <div className="metric-grid compact"><Metric label="Accuracy" value={`${(model.metrics.accuracy*100).toFixed(1)}%`} detail="Correct holdout predictions" icon={<Check size={17}/>}/><Metric label="Macro F1" value={`${(model.metrics.f1_macro*100).toFixed(1)}%`} detail="Average across target classes" icon={<Gauge size={17}/>} accent="violet"/><Metric label="Macro precision" value={`${(model.metrics.precision_macro*100).toFixed(1)}%`} detail="Average across target classes" icon={<Activity size={17}/>} accent="teal"/><Metric label={model.metrics.roc_auc===null?"Holdout rows":"ROC-AUC"} value={model.metrics.roc_auc===null?model.test_rows.toLocaleString():model.metrics.roc_auc.toFixed(3)} detail={model.metrics.roc_auc===null?"Evaluated test examples":`Positive class: ${model.metrics.roc_auc_positive_class}`} icon={<BarChart3 size={17}/>} accent="amber"/></div>
        <p className="result-footnote">{model.train_rows.toLocaleString()} training rows · {model.test_rows.toLocaleString()} test rows · {model.dropped_missing_target.toLocaleString()} missing-target rows and {model.dropped_missing_date.toLocaleString()} rows without a valid selected date excluded. These are evaluation metrics, not a guarantee of future performance.</p>
        <ClassMetricsTable model={model}/>
        <div className="table-wrap prediction-sample"><table><thead><tr><th>Holdout row</th><th>Actual</th><th>Predicted</th><th>Max class probability</th></tr></thead><tbody>{model.sample_predictions.map(row=><tr key={row.row}><td>{row.row}</td><td>{row.actual}</td><td>{row.predicted}</td><td>{(row.max_class_probability*100).toFixed(1)}%</td></tr>)}</tbody></table></div>
      </section>
      <section className="panel section-panel inference-panel">
        <div className="panel-head"><div><h3>Predict from feature values</h3><p>Enter values for the same selected features used during training.</p></div></div>
        <form className="inference-form" onSubmit={event=>{event.preventDefault();void onPredict(values);}}>
          {model.feature_schema.map(feature=><label className="field-control" key={feature.name}>{feature.name}<input required type={feature.kind==="number"?"number":"text"} step={feature.kind==="number"?"any":undefined} value={values[feature.name]||""} placeholder={feature.examples.join(", ")||`Enter ${feature.name}`} onChange={event=>setValues(current=>({...current,[feature.name]:event.target.value}))}/></label>)}
          <button className="primary-btn" type="submit">Generate prediction</button>
        </form>
        {prediction&&<div className="prediction-output"><div><span className="eyebrow">PREDICTED {prediction.target_column.toUpperCase()}</span><strong>{prediction.predicted_class}</strong><span>{(prediction.max_class_probability*100).toFixed(1)}% max class probability · not calibrated</span></div><div className="probability-list">{prediction.probabilities.map(item=><div key={item.label}><span>{item.label}</span><b>{(item.probability*100).toFixed(1)}%</b></div>)}</div></div>}
      </section>
    </>}
  </>;
}
function ClassMetricsTable({model}:{model:ModelResult}) {
  return <div className="table-wrap class-metrics"><table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>{model.class_metrics.map(row=><tr key={row.label}><td className="field-name">{row.label}</td><td>{(row.precision*100).toFixed(1)}%</td><td>{(row.recall*100).toFixed(1)}%</td><td>{(row.f1*100).toFixed(1)}%</td><td>{row.support.toLocaleString()}</td></tr>)}</tbody></table></div>;
}
function ModelPerformancePage({model,onNavigate}:{model:ModelResult|null;onNavigate:(page:string)=>void}) {
  if(!model) return <><PageHeading eyebrow="EVALUATION LAB" title="Model performance" subtitle="Train a model on your uploaded dataset to populate this page with measured holdout results."/><div className="status-hero"><div className="status-hero-icon"><Gauge/></div><div><span className="status-pill pending">NO EVALUATION YET</span><h2>Training output will appear here</h2><p>Select a target and explicit feature columns on Order prediction. The page will then show measured metrics, per-class scores, and held-out predictions.</p><button className="primary-btn" onClick={()=>onNavigate("predict")}>Configure model <ArrowUpRight size={14}/></button></div></div></>;
  return <><PageHeading eyebrow="EVALUATION LAB" title="Model performance" subtitle={`${model.model_name} evaluated on held-out data for ${model.target_column}.`}/><section className="panel section-panel"><div className="panel-head"><div><h3>Classification results</h3><p>{model.split_method} · {model.train_rows.toLocaleString()} train / {model.test_rows.toLocaleString()} test rows</p></div><span className="panel-tag">MEASURED OUTPUT</span></div><div className="metric-grid compact"><Metric label="Accuracy" value={`${(model.metrics.accuracy*100).toFixed(1)}%`} detail="Holdout score" icon={<Check size={17}/>}/><Metric label="Macro precision" value={`${(model.metrics.precision_macro*100).toFixed(1)}%`} detail="Across classes" icon={<Activity size={17}/>} accent="teal"/><Metric label="Macro recall" value={`${(model.metrics.recall_macro*100).toFixed(1)}%`} detail="Across classes" icon={<ArrowUpRight size={17}/>} accent="violet"/><Metric label="Macro F1" value={`${(model.metrics.f1_macro*100).toFixed(1)}%`} detail={model.metrics.roc_auc===null?"Across classes":`ROC-AUC ${model.metrics.roc_auc.toFixed(3)}`} icon={<Gauge size={17}/>} accent="amber"/></div><ClassMetricsTable model={model}/><div className="confusion-grid" style={{"--confusion-columns":model.classes.length} as React.CSSProperties}><div className="confusion-corner">Actual ↓ / Predicted →</div>{model.classes.map(label=><b key={label}>{label}</b>)}{model.confusion_matrix.map((row,index)=><Fragment key={model.classes[index]}><b>{model.classes[index]}</b>{row.map((count,column)=><span key={`${index}-${column}`}>{count.toLocaleString()}</span>)}</Fragment>)}</div></section><section className="panel section-panel"><div className="panel-head"><div><h3>Sample holdout predictions</h3><p>Actual outcomes compared with model predictions · first 20 test rows</p></div></div><div className="table-wrap"><table><thead><tr><th>Row</th><th>Actual</th><th>Predicted</th><th>Max class probability · uncalibrated</th></tr></thead><tbody>{model.sample_predictions.map(row=><tr key={row.row}><td>{row.row}</td><td>{row.actual}</td><td>{row.predicted}</td><td>{(row.max_class_probability*100).toFixed(1)}%</td></tr>)}</tbody></table></div></section></>;
}
function ExplainabilityPage({model,explanation,onExplain,onNavigate}:{model:ModelResult|null;explanation:ExplanationResult|null;onExplain:(values:Record<string,string>)=>Promise<void>;onNavigate:(page:string)=>void}) {
  const [values,setValues] = useState<Record<string,string>>({});
  useEffect(()=>{
    setValues(Object.fromEntries((model?.feature_schema||[]).map(feature=>[feature.name,feature.kind==="number"?"":(feature.examples[0]||"")])));
  },[model]);
  if(!model) return <><PageHeading eyebrow="INTERPRETABILITY" title="Model explainability" subtitle="Inspect global feature importance and an order-level what-if explanation."/><div className="status-hero"><div className="status-hero-icon"><BrainCircuit/></div><div><span className="status-pill pending">MODEL REQUIRED</span><h2>Train a model to explain its outputs</h2><p>Explanations are only available for the model trained on the current CSV and selected features.</p><button className="primary-btn" onClick={()=>onNavigate("predict")}>Configure model <ArrowUpRight size={14}/></button></div></div></>;
  return <>
    <PageHeading eyebrow="INTERPRETABILITY" title="Model explainability" subtitle={`Inspect the Random Forest trained to predict ${model.target_column}. Explanations describe model behavior, not real-world causes.`}/>
    <section className="panel section-panel">
      <div className="panel-head"><div><h3>Global feature importance</h3><p>Which selected fields the fitted forest used most across its trees</p></div><span className="panel-tag">MODEL-LEVEL</span></div>
      <p className="result-footnote">{model.feature_importance_method}</p>
      {model.feature_importance?.length?<div className="chart-area tall"><ResponsiveContainer width="100%" height="100%"><BarChart data={model.feature_importance.slice(0,15)} layout="vertical" margin={{left:12,right:18}}><CartesianGrid stroke="#202d40" horizontal={false}/><XAxis type="number" tickFormatter={value=>`${(Number(value)*100).toFixed(0)}%`} tick={{fill:"#7e8da4",fontSize:10}} axisLine={false} tickLine={false}/><YAxis type="category" dataKey="feature" width={150} tick={{fill:"#9aa9bd",fontSize:9}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#101b2b",border:"1px solid #26364c",borderRadius:10}} formatter={value=>[`${(Number(value)*100).toFixed(2)}%`,"Impurity importance"]}/><Bar dataKey="importance" name="Importance" fill="#8b78df" radius={[0,4,4,0]}/></BarChart></ResponsiveContainer></div>:<div className="empty-chart">No importance values were returned for this model.</div>}
      <p className="result-footnote">Importance does not show whether a field increases or decreases delay risk, is not causal, and can favor high-cardinality fields. See the local what-if sensitivity below.</p>
    </section>
    <section className="panel section-panel">
      <div className="panel-head"><div><h3>Local what-if sensitivity</h3><p>Enter one record using the same inputs selected during training.</p></div><span className="panel-tag">SINGLE-RECORD</span></div>
      <form className="inference-form" onSubmit={event=>{event.preventDefault();void onExplain(values);}}>
        {model.feature_schema.map(feature=><label className="field-control" key={feature.name}>{feature.name}<input required type={feature.kind==="number"?"number":"text"} step={feature.kind==="number"?"any":undefined} value={values[feature.name]||""} placeholder={feature.examples.join(", ")||`Enter ${feature.name}`} onChange={event=>setValues(current=>({...current,[feature.name]:event.target.value}))}/></label>)}
        <button className="primary-btn" type="submit">Explain this prediction</button>
      </form>
      {explanation&&<div className="explanation-result">
        <div className="prediction-output"><div><span className="eyebrow">MODEL PREDICTS {explanation.target_column.toUpperCase()}</span><strong>{explanation.predicted_class}</strong><span>{(explanation.predicted_probability*100).toFixed(1)}% model probability · uncalibrated</span></div><p>{explanation.method} {explanation.baseline}</p></div>
        <div className="chart-area tall"><ResponsiveContainer width="100%" height="100%"><BarChart data={explanation.features} layout="vertical" margin={{left:12,right:18}}><CartesianGrid stroke="#202d40" horizontal={false}/><XAxis type="number" tickFormatter={value=>`${(Number(value)*100).toFixed(0)} pp`} tick={{fill:"#7e8da4",fontSize:10}} axisLine={false} tickLine={false}/><YAxis type="category" dataKey="feature" width={150} tick={{fill:"#9aa9bd",fontSize:9}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#101b2b",border:"1px solid #26364c",borderRadius:10}} formatter={value=>[`${(Number(value)*100).toFixed(2)} pp`,"Probability change"]}/><Bar dataKey="delta" name="Change vs baseline" radius={[0,4,4,0]}>{explanation.features.map(item=><Cell key={item.feature} fill={item.delta>=0?"#d7886a":"#54b89a"}/>)}</Bar></BarChart></ResponsiveContainer></div>
        <div className="table-wrap"><table><thead><tr><th>Feature</th><th>Entered value</th><th>Training baseline</th><th>Probability at value</th><th>At baseline</th><th>Difference</th></tr></thead><tbody>{explanation.features.map(item=><tr key={item.feature}><td className="field-name">{item.feature}</td><td>{item.value}</td><td>{item.baseline_value}</td><td>{(item.probability_at_value*100).toFixed(1)}%</td><td>{(item.probability_at_baseline*100).toFixed(1)}%</td><td>{item.delta>=0?"+":""}{(item.delta*100).toFixed(1)} pp</td></tr>)}</tbody></table></div>
      </div>}
    </section>
  </>;
}
function RobustnessPage({model,onNavigate}:{model:ModelResult|null;onNavigate:(page:string)=>void}) {
  const [measure,setMeasure] = useState<"balanced_accuracy_delta"|"prediction_flip_rate">("balanced_accuracy_delta");
  if(!model) return <><PageHeading eyebrow="RESEARCH VALIDATION" title="Robustness" subtitle="Measure how model predictions change when a selected input is missing at inference time."/><div className="status-hero"><div className="status-hero-icon"><ShieldCheck/></div><div><span className="status-pill pending">MODEL REQUIRED</span><h2>Train a model to run robustness checks</h2><p>The missing-input stress test is calculated on a fixed sample of the held-out rows.</p><button className="primary-btn" onClick={()=>onNavigate("predict")}>Configure model <ArrowUpRight size={14}/></button></div></div></>;
  const robustness=model.robustness;
  const tests=[...(robustness?.tests||[])].sort((a,b)=>a.balanced_accuracy_delta-b.balanced_accuracy_delta);
  const chartData=tests.map(item=>({...item,value:measure==="balanced_accuracy_delta"?item.balanced_accuracy_delta*100:item.prediction_flip_rate*100}));
  return <>
    <PageHeading eyebrow="RESEARCH VALIDATION" title="Robustness" subtitle="Stress-test each selected input by replacing it with missing data on held-out examples."/>
    <div className="notice"><ShieldCheck size={16}/><div><b>Controlled missingness experiment</b><span>{robustness?.method||"No robustness result was returned."} · The model's trained imputer handles the missing value. This is a sensitivity test, not a future-performance guarantee.</span></div></div>
    {robustness&&<>
      <div className="metric-grid compact">
        <Metric label="Stress-test rows" value={robustness.sample_rows.toLocaleString()} detail="Deterministic sample from holdout" icon={<Database size={17}/>}/>
        <Metric label="Baseline accuracy" value={`${(robustness.baseline_accuracy*100).toFixed(1)}%`} detail="Same stress-test sample" icon={<Check size={17}/>} accent="teal"/>
        <Metric label="Balanced accuracy" value={`${(robustness.baseline_balanced_accuracy*100).toFixed(1)}%`} detail="Same stress-test sample" icon={<Gauge size={17}/>} accent="violet"/>
        <Metric label="Inputs tested" value={robustness.tests.length.toLocaleString()} detail="Each feature masked individually" icon={<Activity size={17}/>} accent="amber"/>
      </div>
      <section className="panel section-panel">
        <div className="panel-head"><div><h3>Input-missingness sensitivity</h3><p>Each selected feature is masked individually; results use the same held-out sample.</p></div><label className="chart-select">View<select value={measure} onChange={event=>setMeasure(event.target.value as "balanced_accuracy_delta"|"prediction_flip_rate")}><option value="balanced_accuracy_delta">Balanced accuracy change</option><option value="prediction_flip_rate">Prediction flip rate</option></select></label></div>
        <p className="result-footnote">{measure==="balanced_accuracy_delta"?"Negative values indicate performance loss when that field is missing.":"Share of held-out predictions that changed when that field was missing."}</p>
        <div className="chart-area tall"><ResponsiveContainer width="100%" height="100%"><BarChart data={chartData} layout="vertical" margin={{left:12,right:18}}><CartesianGrid stroke="#202d40" horizontal={false}/><XAxis type="number" tickFormatter={value=>`${Number(value).toFixed(0)}%`} tick={{fill:"#7e8da4",fontSize:10}} axisLine={false} tickLine={false}/><YAxis type="category" dataKey="feature" width={150} tick={{fill:"#9aa9bd",fontSize:9}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#101b2b",border:"1px solid #26364c",borderRadius:10}} formatter={value=>[`${(Number(value)).toFixed(1)}%`,measure==="balanced_accuracy_delta"?"Balanced accuracy change":"Prediction flip rate"]}/><Bar dataKey="value" name={measure==="balanced_accuracy_delta"?"Balanced accuracy change":"Prediction flip rate"} fill="#dd9c55" radius={[0,4,4,0]}/></BarChart></ResponsiveContainer></div>
        <div className="table-wrap"><table><thead><tr><th>Masked feature</th><th>Accuracy</th><th>Balanced accuracy</th><th>Change vs baseline</th><th>Prediction flip rate</th></tr></thead><tbody>{tests.map(item=><tr key={item.feature}><td className="field-name">{item.feature}</td><td>{(item.accuracy*100).toFixed(1)}%</td><td>{(item.balanced_accuracy*100).toFixed(1)}%</td><td>{item.balanced_accuracy_delta>=0?"+":""}{(item.balanced_accuracy_delta*100).toFixed(1)} pp</td><td>{(item.prediction_flip_rate*100).toFixed(1)}%</td></tr>)}</tbody></table></div>
      </section>
    </>}
  </>;
}
function EmptyState({title,text,onUpload}:{title:string;text:string;onUpload:()=>void}) {
  return <div className="empty-state"><div className="empty-icon"><FileSpreadsheet size={25}/></div><h2>{title}</h2><p>{text}</p><button className="primary-btn" onClick={onUpload}><UploadCloud size={16}/> Upload CSV</button></div>;
}
export default App;
