import React, { useState, useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import { 
  Play, Pause, Settings, Cpu, BookOpen, 
  Activity, Terminal, AlertTriangle, CheckCircle, RefreshCw, Upload, Search
} from 'lucide-react';
import './App.css';

function Plot({ data, layout, config, style, onClick }: { data: any[], layout: any, config?: any, style?: React.CSSProperties, onClick?: (event: any) => void }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const onClickRef = useRef(onClick);
  const handleClickRef = useRef<((eventData: any) => void) | null>(null);

  useEffect(() => {
    onClickRef.current = onClick;
  }, [onClick]);

  if (!handleClickRef.current) {
    handleClickRef.current = (eventData: any) => {
      if (onClickRef.current) {
        onClickRef.current(eventData);
      }
    };
  }

  useEffect(() => {
    if (!containerRef.current) return;
    const element = containerRef.current as any;

    Plotly.react(element, data, layout, config).then((el: any) => {
      if (el && typeof el.on === 'function') {
        try {
          el.removeListener('plotly_click', handleClickRef.current!);
        } catch (e) {}
        el.on('plotly_click', handleClickRef.current!);
      }
    });
  }, [data, layout, config]);

  useEffect(() => {
    if (!containerRef.current) return;
    const element = containerRef.current as any;

    const resizeObserver = new ResizeObserver(() => {
      if (element && typeof element.on === 'function') {
        try {
          Plotly.Plots.resize(element);
        } catch (e) {}
      }
    });

    resizeObserver.observe(element);

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  useEffect(() => {
    const element = containerRef.current;
    return () => {
      if (element) {
        try {
          (element as any).removeListener('plotly_click', handleClickRef.current!);
        } catch (e) {}
        Plotly.purge(element);
      }
    };
  }, []);

  return <div ref={containerRef} style={style} />;
}



interface AgentConfig {
  id?: string;
  name: string;
  domain: string;
  knowledge_graph_paths: string[];
  system_prompt_template: string;
  interaction_style: string;
  color: string;
}

interface PhysicsWeights {
  separation: number;
  alignment: number;
  cohesion: number;
  goal_seeking: number;
}

interface PhysicsSettings {
  visual_range: number;
  max_speed: number;
  influence_decay_alpha: number;
  weights: PhysicsWeights;
  dt: number;
}

interface GlobalSettings {
  physics: PhysicsSettings;
  llm_provider: string;
  model_name: string;
  google_api_key?: string;
}

interface LogEntry {
  timestamp: string;
  level: string;
  source: string;
  message: string;
}

interface DOABStatus {
  state: string;
  progress: number;
  status: string;
  error: string;
  result: string;
}

interface GlobalSynthStatus {
  state: string;
  progress: number;
  status: string;
  results: string[];
}

const API_BASE = window.location.port === '5173' ? 'http://localhost:8501' : window.location.origin;

export default function App() {
  const [activeTab, setActiveTab] = useState<'forge' | 'simulation' | 'inspector' | 'kg_viewer' | 'console'>('forge');
  const [settings, setSettings] = useState<GlobalSettings | null>(null);
  const [agents, setAgents] = useState<AgentConfig[]>([]);
  const [kgs, setKgs] = useState<string[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  
  // Sidebar states
  const [llmProvider, setLlmProvider] = useState('ollama');
  const [modelName, setModelName] = useState('gemma4:e2b');
  
  // Invalidation states
  const [lastUpdate, setLastUpdate] = useState(Date.now());

  useEffect(() => {
    fetchSettings();
    fetchAgents();
    fetchKGs();
    fetchLogs();
  }, [lastUpdate]);

  // Log auto-polling
  useEffect(() => {
    const timer = setInterval(() => {
      fetchLogs();
    }, 3000);
    return () => clearInterval(timer);
  }, []);

  const fetchSettings = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/settings`);
      const data = await res.json();
      setSettings(data);
      setLlmProvider(data.llm_provider);
      setModelName(data.model_name);
    } catch (e) {
      console.error('Error fetching settings:', e);
    }
  };

  const fetchAgents = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/agents`);
      const data = await res.json();
      setAgents(data);
    } catch (e) {
      console.error('Error fetching agents:', e);
    }
  };

  const fetchKGs = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/kgs`);
      const data = await res.json();
      setKgs(data);
    } catch (e) {
      console.error('Error fetching KGs:', e);
    }
  };

  const fetchLogs = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/logs`);
      const data = await res.json();
      setLogs(data);
    } catch (e) {
      console.error('Error fetching logs:', e);
    }
  };

  const handleProviderChange = async (provider: string) => {
    setLlmProvider(provider);
    const defaults: Record<string, string> = {
      ollama: 'gemma4:e2b',
      nvidia: 'moonshotai/kimi-k2.5',
      gemini: 'gemini-2.5-flash'
    };
    const suggested = defaults[provider] || modelName;
    setModelName(suggested);
    
    // Apply changes
    await fetch(`${API_BASE}/api/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ llm_provider: provider, model_name: suggested })
    });
    setLastUpdate(Date.now());
  };

  const handleModelChange = async (model: string) => {
    setModelName(model);
    await fetch(`${API_BASE}/api/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ llm_provider: llmProvider, model_name: model })
    });
    setLastUpdate(Date.now());
  };

  const handlePhysicsChange = async (key: string, val: number) => {
    if (!settings) return;
    
    const weights = { ...settings.physics.weights };
    let visual_range = settings.physics.visual_range;
    let max_speed = settings.physics.max_speed;

    if (key === 'separation') weights.separation = val;
    else if (key === 'alignment') weights.alignment = val;
    else if (key === 'cohesion') weights.cohesion = val;
    else if (key === 'goal_seeking') weights.goal_seeking = val;
    else if (key === 'visual_range') visual_range = val;
    else if (key === 'max_speed') max_speed = val;

    // Optimistically update local settings
    const updated = {
      ...settings,
      physics: {
        ...settings.physics,
        visual_range,
        max_speed,
        weights
      }
    };
    setSettings(updated);

    // Call backend API (throttling weights saves is handled nicely)
    await fetch(`${API_BASE}/api/settings/physics`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        separation: weights.separation,
        alignment: weights.alignment,
        cohesion: weights.cohesion,
        goal_seeking: weights.goal_seeking,
        visual_range,
        max_speed
      })
    });
  };

  const saveSettingsToDisk = async () => {
    await fetch(`${API_BASE}/api/settings/save`, { method: 'POST' });
    alert('Settings saved to disk!');
  };

  return (
    <div className="app-container">
      {/* SIDEBAR PANEL */}
      <div className="sidebar">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '1rem', fontWeight: 700, color: '#ffffff' }}>CONFIGURATION</span>
          <button className="btn btn-secondary" style={{ padding: '0.35rem 0.65rem', fontSize: '0.75rem' }} onClick={saveSettingsToDisk}>
            Save Disk
          </button>
        </div>
        
        {/* Model and Provider */}
        <div className="custom-card" style={{ padding: '1rem' }}>
          <div className="card-title" style={{ fontSize: '0.9rem', marginBottom: '0.75rem' }}>
            <Cpu size={16} color="var(--accent-blue)" /> Model & Provider
          </div>
          <div className="input-group" style={{ marginBottom: '0.75rem' }}>
            <label>LLM Provider</label>
            <select value={llmProvider} onChange={(e) => handleProviderChange(e.target.value)}>
              <option value="ollama">Ollama (Local)</option>
              <option value="nvidia">Nvidia NIM</option>
              <option value="gemini">Gemini API</option>
            </select>
          </div>
          <div className="input-group" style={{ marginBottom: '0' }}>
            <label>Model Name</label>
            <input type="text" value={modelName} onChange={(e) => setModelName(e.target.value)} onBlur={(e) => handleModelChange(e.target.value)} />
          </div>
        </div>

        {/* Physics Weight Tuning */}
        <div className="custom-card" style={{ padding: '1.25rem', flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
          <div className="card-title" style={{ fontSize: '0.9rem', marginBottom: '1rem' }}>
            <Settings size={16} color="var(--accent-blue)" /> Physics Flocking Tuner
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', overflowY: 'auto', flexGrow: 1 }}>
            <div className="slider-container">
              <div className="slider-header">
                <span>Separation</span>
                <span className="slider-val">{settings?.physics.weights.separation.toFixed(1) || '1.5'}</span>
              </div>
              <input type="range" min="0" max="5" step="0.1" className="slider-input" 
                value={settings?.physics.weights.separation ?? 1.5} 
                onChange={(e) => handlePhysicsChange('separation', parseFloat(e.target.value))} />
            </div>

            <div className="slider-container">
              <div className="slider-header">
                <span>Alignment</span>
                <span className="slider-val">{settings?.physics.weights.alignment.toFixed(1) || '1.0'}</span>
              </div>
              <input type="range" min="0" max="5" step="0.1" className="slider-input" 
                value={settings?.physics.weights.alignment ?? 1.0} 
                onChange={(e) => handlePhysicsChange('alignment', parseFloat(e.target.value))} />
            </div>

            <div className="slider-container">
              <div className="slider-header">
                <span>Cohesion</span>
                <span className="slider-val">{settings?.physics.weights.cohesion.toFixed(1) || '1.0'}</span>
              </div>
              <input type="range" min="0" max="5" step="0.1" className="slider-input" 
                value={settings?.physics.weights.cohesion ?? 1.0} 
                onChange={(e) => handlePhysicsChange('cohesion', parseFloat(e.target.value))} />
            </div>

            <div className="slider-container">
              <div className="slider-header">
                <span>Goal Seeking</span>
                <span className="slider-val">{settings?.physics.weights.goal_seeking.toFixed(1) || '0.8'}</span>
              </div>
              <input type="range" min="0" max="5" step="0.1" className="slider-input" 
                value={settings?.physics.weights.goal_seeking ?? 0.8} 
                onChange={(e) => handlePhysicsChange('goal_seeking', parseFloat(e.target.value))} />
            </div>

            <div className="divider" style={{ margin: '0.5rem 0' }}></div>

            <div className="slider-container">
              <div className="slider-header">
                <span>Visual Range</span>
                <span className="slider-val">{settings?.physics.visual_range || '25'}</span>
              </div>
              <input type="range" min="1" max="100" className="slider-input" 
                value={settings?.physics.visual_range ?? 25} 
                onChange={(e) => handlePhysicsChange('visual_range', parseInt(e.target.value))} />
            </div>

            <div className="slider-container">
              <div className="slider-header">
                <span>Max Speed</span>
                <span className="slider-val">{settings?.physics.max_speed || '5'}</span>
              </div>
              <input type="range" min="1" max="20" step="0.5" className="slider-input" 
                value={settings?.physics.max_speed ?? 5} 
                onChange={(e) => handlePhysicsChange('max_speed', parseFloat(e.target.value))} />
            </div>
          </div>
        </div>
      </div>

      {/* MAIN CONTENT DASHBOARD */}
      <div className="main-content">
        <div className="header-bar">
          <div className="logo-section">
            <span className="logo-icon">🐝</span>
            <span className="logo-text">Control Tower</span>
          </div>
          
          <div className="tabs-navigation">
            <button className={`tab-button ${activeTab === 'forge' ? 'active' : ''}`} onClick={() => setActiveTab('forge')}>
              Agent Forge
            </button>
            <button className={`tab-button ${activeTab === 'simulation' ? 'active' : ''}`} onClick={() => setActiveTab('simulation')}>
              Swarm Simulation
            </button>
            <button className={`tab-button ${activeTab === 'inspector' ? 'active' : ''}`} onClick={() => setActiveTab('inspector')}>
              Inspector
            </button>
            <button className={`tab-button ${activeTab === 'kg_viewer' ? 'active' : ''}`} onClick={() => setActiveTab('kg_viewer')}>
              Knowledge Graph
            </button>
            <button className={`tab-button ${activeTab === 'console' ? 'active' : ''}`} onClick={() => setActiveTab('console')}>
              Console
            </button>
          </div>
        </div>

        <div className="tab-content">
          <div style={{ display: activeTab === 'forge' ? 'block' : 'none', height: '100%' }}>
            <AgentForge TabCallback={() => setLastUpdate(Date.now())} agents={agents} kgs={kgs} />
          </div>
          <div style={{ display: activeTab === 'simulation' ? 'block' : 'none', height: '100%' }}>
            <SwarmSimulation agents={agents} kgs={kgs} />
          </div>
          <div style={{ display: activeTab === 'inspector' ? 'block' : 'none', height: '100%' }}>
            <Inspector agents={agents} />
          </div>
          <div style={{ display: activeTab === 'kg_viewer' ? 'block' : 'none', height: '100%' }}>
            <KGViewer agents={agents} kgs={kgs} />
          </div>
          <div style={{ display: activeTab === 'console' ? 'block' : 'none', height: '100%' }}>
            <DeveloperConsole logs={logs} onClear={() => fetchLogs()} />
          </div>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------
   TAB Component: Agent Forge
---------------------------------------------------------------- */
function AgentForge({ TabCallback, agents, kgs }: { TabCallback: () => void, agents: AgentConfig[], kgs: string[] }) {
  const [subTab, setSubTab] = useState<'lab' | 'workshop'>('lab');
  
  // KG Architect Form
  const [kgDomain, setKgDomain] = useState('');
  const [kgSourceMode, setKgSourceMode] = useState<'upload' | 'doab'>('upload');
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [doabPrompt, setDoabPrompt] = useState('');
  const [maxPdfs, setMaxPdfs] = useState(3);
  
  // Background task polling states
  const [doabTask, setDoabTask] = useState<DOABStatus | null>(null);
  
  // Agent Workshop Form
  const [mode, setMode] = useState<'create' | 'edit'>('create');
  const [selectedAgentId, setSelectedAgentId] = useState('');
  const [name, setName] = useState('');
  const [domain, setDomain] = useState('');
  const [color, setColor] = useState('#00CCFF');
  const [style, setStyle] = useState('Collaborative');
  const [sysPrompt, setSysPrompt] = useState('');
  const [assignedKGs, setAssignedKGs] = useState<string[]>([]);
  const [enhancingPrompt, setEnhancingPrompt] = useState(false);
  const [savingAgent, setSavingAgent] = useState(false);
  
  // LTM Synthesis states
  const [synthTask, setSynthTask] = useState<GlobalSynthStatus | null>(null);

  // Poll DOAB pipeline
  useEffect(() => {
    let doabTimer: any;
    if (doabTask && (doabTask.state === 'RUNNING' || doabTask.state === 'started')) {
      doabTimer = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/api/kgs/doab/status`);
          const status = await res.json();
          setDoabTask(status);
          if (status.state === 'SUCCESS' || status.state === 'ERROR') {
            TabCallback();
          }
        } catch (e) {
          console.error(e);
        }
      }, 1500);
    }
    return () => clearInterval(doabTimer);
  }, [doabTask]);

  // Poll Global LTM synth task
  useEffect(() => {
    let synthTimer: any;
    if (synthTask && (synthTask.state === 'RUNNING' || synthTask.state === 'started')) {
      synthTimer = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/api/agents/ltm/synthesize-all/status`);
          const status = await res.json();
          setSynthTask(status);
        } catch (e) {
          console.error(e);
        }
      }, 1500);
    }
    return () => clearInterval(synthTimer);
  }, [synthTask]);


  const handleBuildKG = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!kgDomain) return alert('Please specify a domain name');
    
    if (kgSourceMode === 'upload') {
      if (!selectedFiles || selectedFiles.length === 0) return alert('Please upload at least one file');
      
      const formData = new FormData();
      formData.append('domain', kgDomain);
      for (let i = 0; i < selectedFiles.length; i++) {
        formData.append('files', selectedFiles[i]);
      }
      
      try {
        const res = await fetch(`${API_BASE}/api/kgs/build`, {
          method: 'POST',
          body: formData
        });
        const result = await res.json();
        if (res.ok) {
          alert(`Knowledge Graph built successfully: ${result.filename}`);
          TabCallback();
          setKgDomain('');
          setSelectedFiles(null);
        } else {
          alert(`Error building KG: ${result.detail}`);
        }
      } catch (err) {
        console.error(err);
      }
    } else {
      if (!doabPrompt) return alert('Please provide a search prompt');
      
      try {
        setDoabTask({ state: 'started', progress: 0, status: 'Triggering background worker...', error: '', result: '' });
        await fetch(`${API_BASE}/api/kgs/doab`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: doabPrompt, domain: kgDomain, limit: maxPdfs })
        });
      } catch (err) {
        console.error(err);
        setDoabTask(null);
      }
    }
  };

  const handleEditSelectChange = (agentId: string) => {
    setSelectedAgentId(agentId);
    const ag = agents.find(a => a.id === agentId);
    if (ag) {
      setName(ag.name);
      setDomain(ag.domain);
      setColor(ag.color);
      setStyle(ag.interaction_style);
      setSysPrompt(ag.system_prompt_template);
      setAssignedKGs(ag.knowledge_graph_paths);
    }
  };

  const handleEnhancePrompt = async () => {
    if (assignedKGs.length === 0) return alert('Please assign at least one KG to enhance the prompt');
    setEnhancingPrompt(true);
    try {
      const res = await fetch(`${API_BASE}/api/agents/enrich-prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ system_prompt_template: sysPrompt, kg_path: assignedKGs[0] })
      });
      const data = await res.json();
      if (data.status === 'success') {
        setSysPrompt(data.enriched_prompt);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setEnhancingPrompt(false);
    }
  };

  const handleSaveAgent = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !domain) return alert('Name and Domain Expertise are required');
    setSavingAgent(true);
    try {
      const agentData: Partial<AgentConfig> = {
        name,
        domain,
        color,
        interaction_style: style,
        system_prompt_template: sysPrompt,
        knowledge_graph_paths: assignedKGs
      };
      if (mode === 'edit') {
        agentData.id = selectedAgentId;
      }
      
      const res = await fetch(`${API_BASE}/api/agents`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(agentData)
      });
      if (res.ok) {
        alert('Agent saved successfully!');
        TabCallback();
        // Reset form
        if (mode === 'create') {
          setName('');
          setDomain('');
          setColor('#00CCFF');
          setStyle('Collaborative');
          setSysPrompt('');
          setAssignedKGs([]);
        }
      }
    } catch (e) {
      console.error(e);
    } finally {
      setSavingAgent(false);
    }
  };

  const handleSynthesizeLTM = async (agentName: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/agents/${agentName}/ltm/synthesize`, { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        alert(`LTM synthesized successfully! Generated: ${data.filename}`);
      } else {
        alert(data.detail);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleSynthesizeAllLTM = async () => {
    try {
      setSynthTask({ state: 'started', progress: 0, status: 'Initializing LTM pipeline...', results: [] });
      await fetch(`${API_BASE}/api/agents/ltm/synthesize-all`, { method: 'POST' });
    } catch (err) {
      console.error(err);
      setSynthTask(null);
    }
  };

  return (
    <div className="vertical-flow">
      {/* Sub Tabs navigation */}
      <div style={{ display: 'flex', gap: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>
        <button className={`btn ${subTab === 'lab' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setSubTab('lab')}>
          Knowledge Lab
        </button>
        <button className={`btn ${subTab === 'workshop' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setSubTab('workshop')}>
          Agent Workshop
        </button>
      </div>

      {subTab === 'lab' && (
        <div className="grid-2col">
          {/* Building KGs Panel */}
          <div className="custom-card">
            <div className="card-title">📖 Knowledge Graph Architect</div>
            <form onSubmit={handleBuildKG}>
              <div className="input-group">
                <label>Domain Name</label>
                <input type="text" placeholder="e.g. Neuroscience, QuantumPhysics" value={kgDomain} onChange={(e) => setKgDomain(e.target.value)} />
              </div>
              
              <div className="input-group">
                <label>Source Mode</label>
                <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.25rem' }}>
                  <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                    <input type="radio" checked={kgSourceMode === 'upload'} onChange={() => setKgSourceMode('upload')} /> File Upload
                  </label>
                  <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                    <input type="radio" checked={kgSourceMode === 'doab'} onChange={() => setKgSourceMode('doab')} /> DOAB Database
                  </label>
                </div>
              </div>

              {kgSourceMode === 'upload' ? (
                <div className="input-group">
                  <label>Upload Documents (PDF, MD, TXT)</label>
                  <input type="file" multiple onChange={(e) => setSelectedFiles(e.target.files)} />
                </div>
              ) : (
                <>
                  <div className="input-group">
                    <label>Search Topic Description</label>
                    <textarea rows={3} placeholder="Describe the topic you want to research..." value={doabPrompt} onChange={(e) => setDoabPrompt(e.target.value)} />
                  </div>
                  <div className="input-group">
                    <label>Max Reference Files to Download</label>
                    <input type="number" min={1} max={10} value={maxPdfs} onChange={(e) => setMaxPdfs(parseInt(e.target.value))} />
                  </div>
                </>
              )}

              <button className="btn btn-primary" style={{ width: '100%', marginTop: '0.5rem' }} type="submit" disabled={doabTask?.state === 'RUNNING'}>
                {kgSourceMode === 'upload' ? <><Upload size={16} /> Build Graph</> : <><Search size={16} /> Search & Ingest</>}
              </button>
            </form>

            {doabTask && (
              <div style={{ marginTop: '1.5rem' }}>
                {doabTask.state === 'RUNNING' && (
                  <div className="info-status">
                    <span className="spinner"></span> {doabTask.status}
                    <div className="progress-bar-bg"><div className="progress-bar-fill" style={{ width: `${doabTask.progress}%` }}></div></div>
                  </div>
                )}
                {doabTask.state === 'SUCCESS' && (
                  <div className="success-status">
                    <CheckCircle size={16} /> KG created successfully: <code>{doabTask.result}</code>
                    <button className="btn btn-text btn-secondary" onClick={() => fetch(`${API_BASE}/api/kgs/doab/clear`, {method: 'POST'}).then(() => setDoabTask(null))}>Dismiss</button>
                  </div>
                )}
                {doabTask.state === 'ERROR' && (
                  <div className="error-status">
                    <AlertTriangle size={16} /> {doabTask.error}
                    <button className="btn btn-text btn-secondary" onClick={() => fetch(`${API_BASE}/api/kgs/doab/clear`, {method: 'POST'}).then(() => setDoabTask(null))}>Retry</button>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* List KGs Panel */}
          <div className="custom-card">
            <div className="card-title">📚 Available Knowledge Graphs</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', maxHeight: '380px', overflowY: 'auto' }}>
              {kgs.length === 0 ? (
                <div style={{ gridColumn: 'span 2', textAlign: 'center', color: 'var(--text-muted)', paddingTop: '50px' }}>No Knowledge Graphs found. Ingest document to create one.</div>
              ) : (
                kgs.map((kg) => (
                  <div className="kg-card" key={kg}>
                    <div style={{ fontWeight: 600, color: 'var(--accent-blue)', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span>📊</span> {kg.replace('_KG.json', '').replace('.json', '').replace(/_/g, ' ')}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '4px', wordBreak: 'break-all', fontFamily: 'monospace' }}>
                      {kg}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      {subTab === 'workshop' && (
        <div className="grid-2col">
          {/* Agent workshop details */}
          <div className="custom-card">
            <div className="card-title">🤖 Agent Configuration</div>
            
            <div className="input-group" style={{ marginBottom: '1.5rem' }}>
              <label>Forge Mode</label>
              <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.25rem' }}>
                <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                  <input type="radio" checked={mode === 'create'} onChange={() => { setMode('create'); setName(''); setDomain(''); setColor('#00CCFF'); setStyle('Collaborative'); setSysPrompt(''); setAssignedKGs([]); }} /> Create New
                </label>
                <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                  <input type="radio" checked={mode === 'edit'} onChange={() => setMode('edit')} /> Edit Existing
                </label>
              </div>
            </div>

            {mode === 'edit' && (
              <div className="input-group">
                <label>Select Agent</label>
                <select value={selectedAgentId} onChange={(e) => handleEditSelectChange(e.target.value)}>
                  <option value="">-- Choose Agent --</option>
                  {agents.map(a => <option key={a.id} value={a.id}>{a.name}</option>)}
                </select>
              </div>
            )}

            <form onSubmit={handleSaveAgent}>
              <div style={{ display: 'grid', gridTemplateColumns: '3fr 1fr', gap: '1rem' }}>
                <div className="input-group">
                  <label>Agent Name</label>
                  <input type="text" placeholder="e.g. Structural Engineer" value={name} onChange={(e) => setName(e.target.value)} />
                </div>
                <div className="input-group">
                  <label>Badge Color</label>
                  <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    <input type="color" value={color} onChange={(e) => setColor(e.target.value)} style={{ padding: '0', width: '35px', height: '35px', cursor: 'pointer', border: 'none', background: 'transparent' }} />
                    <span style={{ fontSize: '0.8rem', fontFamily: 'monospace' }}>{color}</span>
                  </div>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="input-group">
                  <label>Domain Expertise</label>
                  <input type="text" placeholder="e.g. Physics, Materials Science" value={domain} onChange={(e) => setDomain(e.target.value)} />
                </div>
                <div className="input-group">
                  <label>Interaction Style</label>
                  <select value={style} onChange={(e) => setStyle(e.target.value)}>
                    <option value="Collaborative">Collaborative</option>
                    <option value="Debater">Debater</option>
                    <option value="Socratic">Socratic</option>
                    <option value="Critic">Critic</option>
                  </select>
                </div>
              </div>

              <div className="input-group">
                <label>Assign Knowledge Graphs</label>
                <select multiple value={assignedKGs} onChange={(e) => setAssignedKGs(Array.from(e.target.selectedOptions, option => option.value))} style={{ height: '80px' }}>
                  {kgs.map(kg => <option key={kg} value={kg}>{kg}</option>)}
                </select>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Hold Cmd/Ctrl to select multiple KGs.</span>
              </div>

              <div className="input-group">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <label>System Prompt Template</label>
                  {assignedKGs.length > 0 && (
                    <button type="button" className="btn btn-text" style={{ fontSize: '0.8rem', padding: '0 0.5rem' }} onClick={handleEnhancePrompt} disabled={enhancingPrompt}>
                      {enhancingPrompt ? 'Enhancing...' : '✨ Auto-Enhance'}
                    </button>
                  )}
                </div>
                <textarea rows={5} placeholder="Define agent behavior, objectives, and parameters..." value={sysPrompt} onChange={(e) => setSysPrompt(e.target.value)} />
              </div>

              <button className="btn btn-primary" style={{ width: '100%', marginTop: '0.5rem' }} type="submit" disabled={savingAgent}>
                {savingAgent ? 'Saving...' : mode === 'edit' ? 'Update Agent' : 'Forge Agent'}
              </button>
            </form>
          </div>

          {/* LTM management Panel */}
          <div className="custom-card" style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="card-title">🧠 Swarm Memory & Synthesis</div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', flexGrow: 1 }}>
              <div>
                <h5 style={{ fontSize: '0.9rem', marginBottom: '0.5rem', color: '#ffffff' }}>Synthesize All Agents</h5>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                  Compile the individual thoughts files of all agents in parallel, generating LTM Knowledge Graphs for each.
                </p>
                <button className="btn btn-secondary" style={{ width: '100%' }} onClick={handleSynthesizeAllLTM} disabled={synthTask?.state === 'RUNNING'}>
                  <RefreshCw size={16} /> Synthesize Swarm LTM
                </button>
                
                {synthTask && (
                  <div style={{ marginTop: '1rem' }}>
                    {synthTask.state === 'RUNNING' && (
                      <div className="info-status">
                        <span className="spinner"></span> {synthTask.status}
                        <div className="progress-bar-bg"><div className="progress-bar-fill" style={{ width: `${synthTask.progress}%` }}></div></div>
                      </div>
                    )}
                    {synthTask.state === 'SUCCESS' && (
                      <div className="success-status">
                        <CheckCircle size={16} /> LTM Synthesis complete!
                        <button className="btn btn-text btn-secondary" onClick={() => fetch(`${API_BASE}/api/agents/ltm/synthesize-all/clear`, {method: 'POST'}).then(() => setSynthTask(null))}>Dismiss</button>
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="divider" style={{ margin: '0.5rem 0' }}></div>

              <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                <h5 style={{ fontSize: '0.9rem', marginBottom: '0.5rem', color: '#ffffff' }}>Individual Agent Synthesis</h5>
                <div style={{ overflowY: 'auto', flexGrow: 1, display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '200px' }}>
                  {agents.length === 0 ? (
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', textAlign: 'center', marginTop: '20px' }}>No forged agents available.</div>
                  ) : (
                    agents.map((ag) => (
                      <div className="agent-row-card" key={ag.id}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <div style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: ag.color }}></div>
                          <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>{ag.name}</div>
                        </div>
                        <button className="btn btn-text btn-secondary" style={{ fontSize: '0.75rem', padding: '2px 8px' }} onClick={() => handleSynthesizeLTM(ag.name)}>
                          Synthesize
                        </button>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------
   TAB Component: Swarm Simulation
---------------------------------------------------------------- */
function SwarmSimulation({ agents, kgs: _kgs }: { agents: AgentConfig[], kgs: string[] }) {
  const [universalPrompt, setUniversalPrompt] = useState('Use your expertise to determine how to develop an evolving swarm of multi agent AI agents to continuously improve and adapt...');
  const [dynamicOrchestration, setDynamicOrchestration] = useState(false);
  const [autoAllocateIterations, setAutoAllocateIterations] = useState(3);
  const [stepsPerPhase, setStepsPerPhase] = useState(2);
  const [iterations, setIterations] = useState(10);
  
  // Custom 2-Pane Selector states
  const [selectedAgents, setSelectedAgents] = useState<string[]>([]);
  const [agentSearch, setAgentSearch] = useState('');
  
  // Running simulation status
  const [activeSimId, setActiveSimId] = useState<string | null>(null);
  const [simState, setSimState] = useState<any>(null);
  const [importing, setImporting] = useState(false);

  // Poll active simulation status
  useEffect(() => {
    let simTimer: any;
    if (activeSimId) {
      simTimer = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/api/simulation/status/${activeSimId}`);
          const status = await res.json();
          setSimState(status);
          if (status.status === 'complete' || status.status === 'error') {
            setActiveSimId(null);
          }
        } catch (e) {
          console.error(e);
        }
      }, 1000);
    }
    return () => clearInterval(simTimer);
  }, [activeSimId]);

  const handleAddAgent = (name: string) => {
    if (!selectedAgents.includes(name)) {
      setSelectedAgents([...selectedAgents, name]);
    }
  };

  const handleRemoveAgent = (name: string) => {
    setSelectedAgents(selectedAgents.filter(n => n !== name));
  };

  const handleRunSimulation = async () => {
    if (!dynamicOrchestration && selectedAgents.length === 0) {
      return alert('Please select at least one agent for the swarm.');
    }
    
    try {
      const payload = {
        universal_prompt: universalPrompt,
        dynamic_orchestration_enabled: dynamicOrchestration,
        selected_agent_names: selectedAgents,
        iterations: iterations,
        auto_allocate_iterations: autoAllocateIterations,
        steps_per_phase: stepsPerPhase
      };
      
      const res = await fetch(`${API_BASE}/api/simulation/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.status === 'success') {
        setActiveSimId(data.simulation_id);
        setSimState({ status: 'initializing', iteration: 0, max_iterations: data.max_iterations, message: 'Initializing...' });
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleImportSim = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    
    setImporting(true);
    const formData = new FormData();
    formData.append('file', files[0]);
    
    try {
      const res = await fetch(`${API_BASE}/api/simulation/import`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (data.status === 'success') {
        setSimState(data.live_state);
        alert('Simulation imported successfully!');
      } else {
        alert(data.detail);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setImporting(false);
    }
  };

  const availableConfigs = agents.filter(a => !selectedAgents.includes(a.name));
  const filteredAvailable = availableConfigs.filter(ag => 
    ag.name.toLowerCase().includes(agentSearch.toLowerCase()) || 
    ag.domain.toLowerCase().includes(agentSearch.toLowerCase())
  );
  const selectedConfigs = agents.filter(a => selectedAgents.includes(a.name));

  return (
    <div className="grid-2col" style={{ gridTemplateColumns: '1.2fr 1.8fr', height: 'calc(100vh - 120px)', overflow: 'hidden' }}>
      
      {/* LEFT: Swarm Control Panel */}
      <div className="custom-card" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', overflowY: 'auto' }}>
        <div className="card-title"><Activity size={18} color="var(--accent-blue)" /> Swarm Control Panel</div>
        
        <div className="input-group">
          <label>Universal Prompt (Global Goal)</label>
          <textarea rows={3} value={universalPrompt} onChange={(e) => setUniversalPrompt(e.target.value)} style={{ fontSize: '0.85rem' }} />
        </div>

        <div className="input-group" style={{ marginBottom: '0.25rem' }}>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
            <input type="checkbox" checked={dynamicOrchestration} onChange={(e) => setDynamicOrchestration(e.target.checked)} /> <b>Enable Dynamic Orchestration</b>
          </label>
        </div>

        {dynamicOrchestration ? (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            <div className="input-group">
              <label>Auto-Allocate Phases</label>
              <input type="number" min={1} max={20} value={autoAllocateIterations} onChange={(e) => setAutoAllocateIterations(parseInt(e.target.value))} />
            </div>
            <div className="input-group">
              <label>Steps per Phase</label>
              <input type="number" min={1} max={10} value={stepsPerPhase} onChange={(e) => setStepsPerPhase(parseInt(e.target.value))} />
            </div>
          </div>
        ) : (
          <div className="vertical-flow" style={{ gap: '0.75rem' }}>
            {/* Custom 2-Pane Selector */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div className="input-group" style={{ marginBottom: '0.25rem' }}>
                <input 
                  type="text" 
                  placeholder="🔍 Search available agents..." 
                  value={agentSearch} 
                  onChange={(e) => setAgentSearch(e.target.value)} 
                  style={{ padding: '0.5rem 0.75rem', fontSize: '0.8rem', background: 'rgba(0, 0, 0, 0.3)' }} 
                />
              </div>

              <div className="pane-container" style={{ marginTop: '0', gap: '0.75rem' }}>
                {/* Available pane */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 'bold', color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between', padding: '0 2px' }}>
                    <span>AVAILABLE AGENTS ({filteredAvailable.length})</span>
                  </div>
                  <div className="pane" style={{ height: '220px', padding: '0.4rem', border: '1px solid var(--border-color)', background: 'rgba(0,0,0,0.2)' }}>
                    {filteredAvailable.length === 0 ? (
                      <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textAlign: 'center', marginTop: '80px' }}>
                        {agentSearch ? 'No matches' : 'No agents'}
                      </div>
                    ) : (
                      filteredAvailable.map(ag => (
                        <div 
                          className="agent-row-card" 
                          key={ag.id} 
                          style={{ 
                            padding: '0.5rem 0.7rem', 
                            cursor: 'pointer', 
                            display: 'flex', 
                            justifyContent: 'space-between', 
                            alignItems: 'center',
                            borderLeft: `4px solid ${ag.color}`
                          }}
                          onClick={() => handleAddAgent(ag.name)}
                          title="Click to add to swarm"
                        >
                          <div className="agent-meta" style={{ paddingLeft: '2px', borderLeft: 'none' }}>
                            <span className="agent-meta-name" style={{ fontSize: '0.8rem', fontWeight: 600 }}>{ag.name}</span>
                            <span className="agent-meta-sub" style={{ fontSize: '0.65rem', opacity: 0.8 }}>{ag.domain}</span>
                          </div>
                          <span style={{ fontSize: '0.85rem', color: 'var(--accent-blue)', opacity: 0.7, padding: '0 4px' }}>➕</span>
                        </div>
                      ))
                    )}
                  </div>
                </div>

                {/* Selected pane */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 'bold', color: 'var(--accent-blue)', display: 'flex', justifyContent: 'space-between', padding: '0 2px' }}>
                    <span>SELECTED SWARM ({selectedConfigs.length})</span>
                  </div>
                  <div className="pane" style={{ height: '220px', padding: '0.4rem', border: '1px solid rgba(0, 242, 254, 0.15)', background: 'rgba(0, 242, 254, 0.02)' }}>
                    {selectedConfigs.length === 0 ? (
                      <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textAlign: 'center', marginTop: '80px' }}>
                        Click agents to add
                      </div>
                    ) : (
                      selectedConfigs.map(ag => (
                        <div 
                          className="agent-row-card" 
                          key={ag.id} 
                          style={{ 
                            padding: '0.5rem 0.7rem', 
                            cursor: 'pointer', 
                            display: 'flex', 
                            justifyContent: 'space-between', 
                            alignItems: 'center',
                            borderLeft: `4px solid ${ag.color}`
                          }}
                          onClick={() => handleRemoveAgent(ag.name)}
                          title="Click to remove from swarm"
                        >
                          <div className="agent-meta" style={{ paddingLeft: '2px', borderLeft: 'none' }}>
                            <span className="agent-meta-name" style={{ fontSize: '0.8rem', fontWeight: 600 }}>{ag.name}</span>
                            <span className="agent-meta-sub" style={{ fontSize: '0.65rem', opacity: 0.8 }}>{ag.domain}</span>
                          </div>
                          <span style={{ fontSize: '0.85rem', color: 'var(--accent-red)', opacity: 0.7, padding: '0 4px' }}>➖</span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </div>

            <div className="input-group" style={{ marginBottom: '0' }}>
              <label>Iterations</label>
              <input type="number" min={1} value={iterations} onChange={(e) => setIterations(parseInt(e.target.value))} />
            </div>
          </div>
        )}

        <button className="btn btn-primary" style={{ width: '100%', marginTop: '0.25rem' }} onClick={handleRunSimulation} disabled={activeSimId !== null}>
          {activeSimId ? <><span className="spinner" style={{ width: '14px', height: '14px' }}></span> Running...</> : 'Initialize & Run Swarm'}
        </button>

        <div className="divider" style={{ margin: '0.25rem 0' }}></div>

        {/* Import Panel */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Import Prior Simulation</span>
          <label className="btn btn-secondary" style={{ padding: '0.4rem 0.75rem', fontSize: '0.75rem', cursor: importing ? 'default' : 'pointer', opacity: importing ? 0.6 : 1 }}>
            {importing ? 'Importing...' : <><Upload size={12} /> Choose JSON</>}
            <input type="file" onChange={handleImportSim} style={{ display: 'none' }} disabled={importing} />
          </label>
        </div>
      </div>

      {/* RIGHT: Swarm Visualization Tower */}
      <div className="custom-card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div className="card-title">📡 Swarm Visualization Tower</div>
        
        <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {simState ? (
            <SimVisualizer simState={simState} universalPrompt={universalPrompt} />
          ) : (
            <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', color: 'var(--text-muted)' }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🌌</div>
              <h4>Control Tower Active</h4>
              <p style={{ fontSize: '0.85rem', width: '320px', textAlign: 'center', marginTop: '0.5rem', lineHeight: 1.5 }}>
                Configure swarm parameters on the left and run simulation to observe murmuration dynamics in 3D space.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------
   Helper Component: SimVisualizer (active & playback states)
---------------------------------------------------------------- */
function SimVisualizer({ simState, universalPrompt }: { simState: any, universalPrompt: string }) {
  const [playbackStep, setPlaybackStep] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const playTimerRef = useRef<any>(null);

  
  // Consensus Synthesis states
  const [generatingPaper, setGeneratingPaper] = useState(false);
  const [generatingKG, setGeneratingKG] = useState(false);
  const [kgDomain, setKgDomain] = useState('Collaborative_KG');
  const [paperResult, setPaperResult] = useState('');
  const [showPaperModal, setShowPaperModal] = useState(false);

  const status = simState.status;
  const message = simState.message || '';
  const visHistory = simState.vis_history || [];
  const history = simState.history || [];
  const maxIterations = simState.max_iterations || 1;
  const currentIteration = simState.iteration || 0;

  // Sync playbackStep to end of simulation when history updates
  useEffect(() => {
    if (status === 'complete' && history.length > 0) {
      setPlaybackStep(history.length - 1);
    }
  }, [status, history]);

  // Handle Playback Interval timer
  useEffect(() => {
    if (isPlaying && history.length > 0) {
      const intervalMs = 120 / playbackSpeed;
      playTimerRef.current = setInterval(() => {
        setPlaybackStep((prev) => {
          if (prev >= history.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, intervalMs);
    } else {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    }
    return () => {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    };
  }, [isPlaying, playbackSpeed, history]);

  const handleGeneratePaper = async () => {
    if (!simState.history || simState.history.length === 0) return;
    setGeneratingPaper(true);
    try {
      const res = await fetch(`${API_BASE}/api/simulation/${simState.simulation_id}/generate-paper`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ universal_prompt: universalPrompt })
      });
      const data = await res.json();
      if (data.status === 'success') {
        setPaperResult(data.paper_content);
        setShowPaperModal(true);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setGeneratingPaper(false);
    }
  };

  const handleGenerateKG = async () => {
    const conclusionText = simState.sim_final_state?.consensus?.enriched_conclusion || '';
    if (!conclusionText) {
      return alert('No consensus text found. Play the simulation to completion first.');
    }
    setGeneratingKG(true);
    try {
      const res = await fetch(`${API_BASE}/api/simulation/${simState.simulation_id}/generate-kg`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conclusion_text: conclusionText, kg_domain: kgDomain })
      });
      const data = await res.json();
      if (data.status === 'success') {
        alert(`Collaborative KG created: ${data.filename}`);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setGeneratingKG(false);
    }
  };

  // Compile Plotly Data
  let plotlyData: any[] = [];
  let isDynamicView = (status === 'running' || status === 'initializing' || status === 'voting' || status === 'saving' || status === 'enriching');
  
  if (isDynamicView) {
    // ACTIVE SIMULATION VIEW
    if (visHistory.length > 0) {
      const latestFrame = visHistory[visHistory.length - 1];
      const agents = latestFrame.agents || [];
      const x = agents.map((a: any) => a.position.x);
      const y = agents.map((a: any) => a.position.y);
      const z = agents.map((a: any) => a.position.z);
      const colors = agents.map((a: any) => a.color);
      const texts = agents.map((a: any) => `${a.name}<br>${a.domain}`);
      
      // Draw trails
      if (visHistory.length > 1) {
        const agentIds = agents.map((a: any) => a.id);
        agentIds.forEach((aid: string, idx: number) => {
          const trailX: number[] = [];
          const trailY: number[] = [];
          const trailZ: number[] = [];
          visHistory.forEach((frame: any) => {
            const a = frame.agents.find((ag: any) => ag.id === aid);
            if (a) {
              trailX.push(a.position.x);
              trailY.push(a.position.y);
              trailZ.push(a.position.z);
            }
          });
          plotlyData.push({
            x: trailX, y: trailY, z: trailZ,
            type: 'scatter3d', mode: 'lines',
            line: { color: colors[idx], width: 2.5 },
            opacity: 0.35, showlegend: false
          });
        });
      }
      
      // Draw agents
      plotlyData.push({
        x, y, z,
        type: 'scatter3d', mode: 'markers',
        marker: { size: 7, color: colors, opacity: 0.9, line: { color: 'rgba(255,255,255,0.2)', width: 0.5 } },
        text: texts, name: 'Agents'
      });
    }
    
    // Draw goal coordinate
    plotlyData.push({
      x: [0], y: [0], z: [0],
      type: 'scatter3d', mode: 'markers',
      marker: { size: 10, color: 'gold', symbol: 'diamond' },
      name: 'Goal'
    });
  } else {
    // SIMULATION PLAYBACK VIEW
    if (history.length > 0 && playbackStep < history.length) {
      const snap = history[playbackStep];
      const agentStates = snap.agent_states || {};
      const agentConfigs = snap.agent_configs || {};
      
      const x: number[] = [];
      const y: number[] = [];
      const z: number[] = [];
      const colors: string[] = [];
      const texts: string[] = [];
      
      Object.keys(agentStates).forEach((aid) => {
        const ast = agentStates[aid];
        const cfg = agentConfigs[ast.config_id];
        if (cfg) {
          x.push(ast.position.x);
          y.push(ast.position.y);
          z.push(ast.position.z);
          colors.push(cfg.color);
          texts.push(`${cfg.name}<br>${cfg.domain}`);
        }
      });
      
      // Draw agents
      plotlyData.push({
        x, y, z,
        type: 'scatter3d', mode: 'markers',
        marker: { size: 7, color: colors, opacity: 0.85 },
        text: texts, name: 'Agents'
      });
      
      // Draw goal coordinate
      plotlyData.push({
        x: [0], y: [0], z: [0],
        type: 'scatter3d', mode: 'markers',
        marker: { size: 10, color: 'gold', symbol: 'diamond' },
        name: 'Goal'
      });
      
      // Draw influence lines
      const ix: (number | null)[] = [];
      const iy: (number | null)[] = [];
      const iz: (number | null)[] = [];
      const influence_matrix = snap.influence_matrix || {};
      const threshold = 0.1;
      
      Object.keys(influence_matrix).forEach((r_id) => {
        if (agentStates[r_id]) {
          const r_pos = agentStates[r_id].position;
          const row = influence_matrix[r_id] || {};
          Object.keys(row).forEach((s_id) => {
            const score = row[s_id];
            if (r_id !== s_id && score > threshold && agentStates[s_id]) {
              const s_pos = agentStates[s_id].position;
              ix.push(r_pos.x, s_pos.x, null);
              iy.push(r_pos.y, s_pos.y, null);
              iz.push(r_pos.z, s_pos.z, null);
            }
          });
        }
      });
      
      if (ix.length > 0) {
        plotlyData.push({
          x: ix, y: iy, z: iz,
          type: 'scatter3d', mode: 'lines',
          line: { color: 'rgba(255, 255, 255, 0.15)', width: 2.5 },
          showlegend: false, name: 'Influence'
        });
      }
    }
  }

  // Visual axis ranges based on history coordinates
  let all_x: number[] = [];
  let all_y: number[] = [];
  let all_z: number[] = [];
  const src_list = isDynamicView ? visHistory : history;
  
  src_list.forEach((snap: any) => {
    const agents = isDynamicView ? (snap.agents || []) : Object.values(snap.agent_states || {});
    agents.forEach((ag: any) => {
      const pos = isDynamicView ? ag.position : ag.position;
      all_x.push(pos.x);
      all_y.push(pos.y);
      all_z.push(pos.z);
    });
  });
  
  const padding = 5;
  const xr = all_x.length > 0 ? [Math.min(...all_x) - padding, Math.max(...all_x) + padding] : [-30, 30];
  const yr = all_y.length > 0 ? [Math.min(...all_y) - padding, Math.max(...all_y) + padding] : [-30, 30];
  const zr = all_z.length > 0 ? [Math.min(...all_z) - padding, Math.max(...all_z) + padding] : [-30, 30];

  const layout = {
    template: 'plotly_dark',
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    scene: {
      xaxis: { range: xr, title: 'X' },
      yaxis: { range: yr, title: 'Y' },
      zaxis: { range: zr, title: 'Z' },
      aspectmode: 'cube'
    },
    margin: { l: 0, r: 0, b: 0, t: 0 },
    height: 340,
    font: { color: 'white' },
    showlegend: false
  };

  const currentSnap = history[playbackStep] || null;
  const agentThoughts = currentSnap ? Object.keys(currentSnap.agent_states || {}).map(aid => {
    const ast = currentSnap.agent_states[aid];
    const cfg = currentSnap.agent_configs[ast.config_id];
    return { name: cfg.name, thought: ast.latest_thought || '...' };
  }) : [];

  const currentThoughts = isDynamicView 
    ? (visHistory.length > 0 ? (visHistory[visHistory.length - 1].agents || []).map((ag: any) => ({ name: ag.name, thought: ag.thought || '...' })) : [])
    : agentThoughts;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flexGrow: 1, overflow: 'hidden' }}>
      {/* Simulation Info Banner */}
      <div style={{ marginBottom: '1rem', padding: '0.75rem', borderRadius: '8px', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <span style={{ fontSize: '0.85rem', fontWeight: 'bold', textTransform: 'uppercase', color: 'var(--accent-blue)' }}>
            STATUS: {status}
          </span>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            {message}
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <span style={{ fontSize: '0.85rem', fontFamily: 'monospace' }}>
            Iteration: {currentIteration}/{maxIterations}
          </span>
          <div className="progress-bar-bg" style={{ width: '120px', height: '4px' }}>
            <div className="progress-bar-fill" style={{ width: `${(currentIteration / maxIterations) * 100}%` }}></div>
          </div>
        </div>
      </div>

      {/* Plotly Canvas Container */}
      <div style={{ height: '340px', background: 'rgba(0,0,0,0.2)', border: '1px solid var(--border-color)', borderRadius: '8px', overflow: 'hidden', position: 'relative', flexShrink: 0 }}>
        {plotlyData.length > 0 ? (
          <Plot data={plotlyData} layout={layout} config={{ responsive: true, displayModeBar: false }} style={{ width: '100%', height: '100%' }} />
        ) : (
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: 'var(--text-muted)' }}>
            Generating Plotly coordinates...
          </div>
        )}
      </div>

      {/* Real-time Agents' Thoughts Pane (during active simulation) */}
      {isDynamicView && (
        <div style={{ display: 'flex', flexDirection: 'column', flexGrow: 1, overflow: 'hidden', marginTop: '1rem' }}>
          <span style={{ fontSize: '0.85rem', fontWeight: 'bold', color: 'var(--text-secondary)', marginBottom: '0.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>💭 AGENTS' THOUGHTS (Real-time)</span>
            <span className="spinner" style={{ width: '12px', height: '12px' }}></span>
          </span>
          <div style={{ overflowY: 'auto', flexGrow: 1, border: '1px solid var(--border-color)', borderRadius: '6px', background: 'rgba(0,0,0,0.15)', padding: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {currentThoughts.length === 0 ? (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textAlign: 'center', padding: '2rem' }}>
                Waiting for agent thoughts...
              </div>
            ) : (
              currentThoughts.map((at: any) => (
                <div key={at.name} style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem', padding: '0.4rem', borderRadius: '4px', background: 'rgba(255,255,255,0.02)', borderLeft: '3px solid rgba(0, 242, 254, 0.4)' }}>
                  <span style={{ fontSize: '0.75rem', fontWeight: 'bold', color: 'var(--accent-blue)' }}>{at.name}</span>
                  <p style={{ fontSize: '0.75rem', color: '#e2e8f0', lineHeight: 1.3 }}>{at.thought}</p>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* Playback Controls & Tabs */}
      {!isDynamicView && history.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', flexGrow: 1, overflow: 'hidden', marginTop: '1rem' }}>
          
          {/* Controls toolbar */}
          <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 2.5fr 1fr 1fr', gap: '1rem', alignItems: 'center', flexShrink: 0, marginBottom: '0.75rem' }}>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button className="btn btn-secondary" onClick={() => setIsPlaying(!isPlaying)} style={{ padding: '0.4rem 0.8rem' }}>
                {isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? 'Pause' : 'Play'}
              </button>
              <select value={playbackSpeed} onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))} style={{ background: 'rgba(0,0,0,0.2)', border: '1px solid var(--border-color)', borderRadius: '6px', color: '#fff', fontSize: '0.8rem', padding: '0.25rem' }}>
                <option value="0.5">0.5x</option>
                <option value="1">1.0x</option>
                <option value="2">2.0x</option>
                <option value="4">4.0x</option>
              </select>
            </div>
            
            <div className="slider-container" style={{ marginBottom: 0 }}>
              <input type="range" min="0" max={history.length - 1} className="slider-input" value={playbackStep} onChange={(e) => setPlaybackStep(parseInt(e.target.value))} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                <span>Step 0</span>
                <span style={{ color: 'var(--accent-blue)', fontWeight: 'bold' }}>Step {playbackStep}</span>
                <span>Step {history.length - 1}</span>
              </div>
            </div>

            <div style={{ fontSize: '0.8rem', textAlign: 'center' }}>
              Total Steps: <span style={{ color: '#fff', fontWeight: 'bold' }}>{history.length}</span>
            </div>

            {status === 'complete' && (
              <button className="btn btn-primary" style={{ padding: '0.4rem 0.65rem', fontSize: '0.8rem' }} onClick={handleGeneratePaper} disabled={generatingPaper}>
                {generatingPaper ? 'Generating...' : 'Consensus Paper'}
              </button>
            )}
          </div>

          {/* Details & Synthesis panel */}
          <div className="divider" style={{ margin: '0.25rem 0' }}></div>
          
          <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: '1rem', flexGrow: 1, overflow: 'hidden' }}>
            {/* Thought Inspector */}
            <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 'bold', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>💭 THOUGHT INSPECTOR (Step {playbackStep})</span>
              <div style={{ overflowY: 'auto', flexGrow: 1, border: '1px solid var(--border-color)', borderRadius: '6px', background: 'rgba(0,0,0,0.15)', padding: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {agentThoughts.map(at => (
                  <div key={at.name} style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem', padding: '0.4rem', borderRadius: '4px', background: 'rgba(255,255,255,0.02)', borderLeft: '3px solid rgba(0, 242, 254, 0.4)' }}>
                    <span style={{ fontSize: '0.75rem', fontWeight: 'bold', color: 'var(--accent-blue)' }}>{at.name}</span>
                    <p style={{ fontSize: '0.75rem', color: '#e2e8f0', lineHeight: 1.3 }}>{at.thought}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Collaborative KG Synthesis Panel */}
            <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', border: '1px solid var(--border-color)', borderRadius: '6px', background: 'rgba(0,0,0,0.15)', padding: '0.75rem' }}>
              <div>
                <span style={{ fontSize: '0.8rem', fontWeight: 'bold', color: '#fff', display: 'block', marginBottom: '0.5rem' }}>📝 Collaborative KG Synthesis</span>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.4, marginBottom: '0.75rem' }}>
                  Extract a collaborative domain Knowledge Graph from the final swarm consensus thoughts.
                </p>
                <div className="input-group" style={{ marginBottom: '0.75rem' }}>
                  <label style={{ fontSize: '0.75rem' }}>New KG Name</label>
                  <input type="text" value={kgDomain} onChange={(e) => setKgDomain(e.target.value)} style={{ padding: '0.4rem', fontSize: '0.8rem' }} />
                </div>
              </div>
              <button className="btn btn-secondary" style={{ width: '100%', fontSize: '0.8rem', padding: '0.5rem' }} onClick={handleGenerateKG} disabled={generatingKG}>
                {generatingKG ? 'Extracting...' : 'Generate Collaborative KG'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Research Paper modal dialog */}
      {showPaperModal && (
        <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', background: 'rgba(0,0,0,0.65)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000, backdropFilter: 'blur(8px)' }}>
          <div className="custom-card" style={{ width: '800px', height: '600px', display: 'flex', flexDirection: 'column', background: 'var(--bg-primary)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h4 style={{ color: '#fff' }}>📄 Generated Swarm Research Paper</h4>
              <button className="btn btn-secondary" style={{ padding: '0.35rem 0.75rem' }} onClick={() => setShowPaperModal(false)}>
                Close
              </button>
            </div>
            <div style={{ flexGrow: 1, overflowY: 'auto', background: 'rgba(0,0,0,0.2)', border: '1px solid var(--border-color)', borderRadius: '6px', padding: '1rem', fontFamily: 'serif', lineHeight: 1.6, color: '#f1f5f9', whiteSpace: 'pre-wrap' }}>
              {paperResult}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------
   TAB Component: Inspector (Pydantic state viewer)
---------------------------------------------------------------- */
function Inspector({ agents }: { agents: AgentConfig[] }) {
  const [selectedAgent, setSelectedAgent] = useState('');
  const [agentData, setAgentData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (selectedAgent) {
      loadAgentDetails();
    } else {
      setAgentData(null);
    }
  }, [selectedAgent]);

  const loadAgentDetails = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/agents`);
      const list = await res.json();
      const details = list.find((a: any) => a.name === selectedAgent);
      setAgentData(details);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="custom-card" style={{ maxWidth: '650px' }}>
      <div className="card-title">🔍 Agent Private Memory Inspector</div>
      <div className="input-group">
        <label>Select Swarm Expert</label>
        <select value={selectedAgent} onChange={(e) => setSelectedAgent(e.target.value)}>
          <option value="">-- Choose Agent --</option>
          {agents.map(a => <option key={a.id} value={a.name}>{a.name}</option>)}
        </select>
      </div>

      <div style={{ marginTop: '1.5rem' }}>
        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '40px' }}><span className="spinner"></span></div>
        ) : agentData ? (
          <pre style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', borderRadius: '6px', padding: '1rem', overflowX: 'auto', fontSize: '0.8rem', fontFamily: 'monospace', color: 'var(--accent-blue)', maxHeight: '380px' }}>
            {JSON.stringify(agentData, null, 2)}
          </pre>
        ) : (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textAlign: 'center', paddingTop: '50px' }}>Select an agent config to inspect parameters and schemas.</div>
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------
   TAB Component: KGViewer (nodes-edges Plotly generator)
---------------------------------------------------------------- */
function KGViewer({ agents, kgs }: { agents: AgentConfig[], kgs: string[] }) {
  const [kgSource, setKgSource] = useState<'global' | 'ltm'>('global');
  const [selectedGlobalKG, setSelectedGlobalKG] = useState('');
  const [selectedAgentKG, setSelectedAgentKG] = useState('');
  const [kgNodesData, setKgNodesData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [nodeDetails, setNodeDetails] = useState<any>(null);

  // Filter agents that have generated LTM KGs
  const agentsWithLTM = agents.filter(a => a.name); // Simple mapping check

  useEffect(() => {
    const filename = kgSource === 'global' ? selectedGlobalKG : `agent_${selectedAgentKG.replace(/ /g, '_')}_ltm_kg.json`;
    if ((kgSource === 'global' && selectedGlobalKG) || (kgSource === 'ltm' && selectedAgentKG)) {
      loadKGContent(filename);
    } else {
      setKgNodesData(null);
    }
  }, [kgSource, selectedGlobalKG, selectedAgentKG]);

  const loadKGContent = async (filename: string) => {
    setLoading(true);
    setNodeDetails(null);
    try {
      const res = await fetch(`${API_BASE}/api/kgs/view/${filename}`);
      const data = await res.json();
      setKgNodesData(data);
    } catch (e) {
      console.error(e);
      setKgNodesData(null);
    } finally {
      setLoading(false);
    }
  };

  // Compile Fibonacci Sphere 3D Node Map
  let plotlyData: any[] = [];
  let nodeKeys: string[] = [];
  
  if (kgNodesData && kgNodesData.nodes) {
    const nodes = kgNodesData.nodes;
    nodeKeys = Object.keys(nodes);
    const count = nodeKeys.length;
    
    if (count > 0) {
      const points: Record<string, number[]> = {};
      const offset = 2.0 / Math.max(1, count);
      const increment = Math.PI * (3.0 - Math.sqrt(5.0)); // golden angle
      
      nodeKeys.forEach((nid, i) => {
        const y = ((i * offset) - 1) + (offset / 2);
        const r = Math.sqrt(1 - Math.pow(y, 2));
        const phi = ((i + 1) % count) * increment;
        
        const scale = 15;
        const x = Math.cos(phi) * r * scale;
        const z = Math.sin(phi) * r * scale;
        const y_scaled = y * scale;
        
        points[nid] = [x, y_scaled, z];
      });

      const xVals = nodeKeys.map(nid => points[nid][0]);
      const yVals = nodeKeys.map(nid => points[nid][1]);
      const zVals = nodeKeys.map(nid => points[nid][2]);
      const labels = nodeKeys.map(nid => nodes[nid].label || nid);

      // Draw Edges
      const ex: (number | null)[] = [];
      const ey: (number | null)[] = [];
      const ez: (number | null)[] = [];
      
      nodeKeys.forEach(nid => {
        const children = nodes[nid].children || [];
        children.forEach((cid: string) => {
          if (points[cid]) {
            ex.push(points[nid][0], points[cid][0], null);
            ey.push(points[nid][1], points[cid][1], null);
            ez.push(points[nid][2], points[cid][2], null);
          }
        });
      });

      if (ex.length > 0) {
        plotlyData.push({
          x: ex, y: ey, z: ez,
          type: 'scatter3d', mode: 'lines',
          line: { color: 'rgba(255,255,255,0.25)', width: 1.5 },
          hoverinfo: 'none', showlegend: false
        });
      }

      // Draw Nodes
      plotlyData.push({
        x: xVals, y: yVals, z: zVals,
        type: 'scatter3d', mode: 'markers',
        marker: { size: 8, color: '#00CCFF', opacity: 0.85 },
        text: labels, hoverinfo: 'text', name: 'Nodes'
      });
    }
  }

  const layout = {
    template: 'plotly_dark',
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    scene: { aspectmode: 'cube' },
    margin: { l: 0, r: 0, b: 0, t: 0 },
    height: 400
  };

  const handlePlotClick = (data: any) => {
    if (!data.points || data.points.length === 0 || !kgNodesData || !kgNodesData.nodes) return;
    
    // Find node clicked by index
    const index = data.points[0].pointNumber;
    if (index >= 0 && index < nodeKeys.length) {
      const nid = nodeKeys[index];
      const node = kgNodesData.nodes[nid];
      
      const connected: string[] = [];
      // Children
      (node.children || []).forEach((c: string) => {
        if (kgNodesData.nodes[c]) connected.push(kgNodesData.nodes[c].label || c);
      });
      // Parents
      Object.keys(kgNodesData.nodes).forEach(pKey => {
        if ((kgNodesData.nodes[pKey].children || []).includes(nid)) {
          connected.push(kgNodesData.nodes[pKey].label || pKey);
        }
      });
      
      setNodeDetails({
        label: node.label || nid,
        summary: node.summary || 'No summary available.',
        connected: Array.from(new Set(connected))
      });
    }
  };

  return (
    <div className="grid-2col" style={{ gridTemplateColumns: '1.2fr 1.8fr' }}>
      {/* Selector panel */}
      <div className="custom-card">
        <div className="card-title"><BookOpen size={18} color="var(--accent-blue)" /> Knowledge Graph Viewer</div>
        
        <div className="input-group" style={{ marginBottom: '1.25rem' }}>
          <label>Graph Source</label>
          <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.25rem' }}>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input type="radio" checked={kgSource === 'global'} onChange={() => setKgSource('global')} /> Global Graphs
            </label>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input type="radio" checked={kgSource === 'ltm'} onChange={() => setKgSource('ltm')} /> Agent Memories
            </label>
          </div>
        </div>

        {kgSource === 'global' ? (
          <div className="input-group">
            <label>Select Global KG</label>
            <select value={selectedGlobalKG} onChange={(e) => setSelectedGlobalKG(e.target.value)}>
              <option value="">-- Choose KG --</option>
              {kgs.map(k => <option key={k} value={k}>{k}</option>)}
            </select>
          </div>
        ) : (
          <div className="input-group">
            <label>Select Agent Memory</label>
            <select value={selectedAgentKG} onChange={(e) => setSelectedAgentKG(e.target.value)}>
              <option value="">-- Choose Agent --</option>
              {agentsWithLTM.map(a => <option key={a.id} value={a.name}>{a.name}</option>)}
            </select>
          </div>
        )}

        {nodeDetails && (
          <div style={{ marginTop: '1.5rem', padding: '1rem', border: '1px solid var(--border-color)', borderRadius: '6px', background: 'rgba(0,0,0,0.2)' }}>
            <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--accent-blue)', fontWeight: 'bold' }}>Node Details</span>
            <h5 style={{ fontSize: '1rem', color: '#fff', margin: '4px 0 8px 0' }}>{nodeDetails.label}</h5>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.4, marginBottom: '0.75rem' }}>{nodeDetails.summary}</p>
            {nodeDetails.connected.length > 0 && (
              <div>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>CONNECTED CONCEPTS:</span>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                  {nodeDetails.connected.map((c: string) => <span key={c} className="badge-pill" style={{ fontSize: '0.65rem' }}>{c}</span>)}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Graph Visualizer */}
      <div className="custom-card" style={{ display: 'flex', flexDirection: 'column' }}>
        <div style={{ flexGrow: 1, height: '400px', background: 'rgba(0,0,0,0.25)', border: '1px solid var(--border-color)', borderRadius: '8px', overflow: 'hidden' }}>
          {loading ? (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}><span className="spinner"></span></div>
          ) : plotlyData.length > 0 ? (
            <Plot data={plotlyData} layout={layout} onClick={handlePlotClick} config={{ responsive: true, displayModeBar: false }} style={{ width: '100%', height: '100%' }} />
          ) : (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              Select a Knowledge Graph to load coordinates and visualize connections.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------
   TAB Component: Developer Console (Logs streaming)
---------------------------------------------------------------- */
function DeveloperConsole({ logs, onClear }: { logs: LogEntry[], onClear: () => void }) {
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [logs]);

  const handleClear = async () => {
    await fetch(`${API_BASE}/api/logs/clear`, { method: 'POST' });
    onClear();
  };

  return (
    <div className="custom-card" style={{ display: 'flex', flexDirection: 'column' }}>
      <div className="title-action-header">
        <div className="card-title" style={{ marginBottom: 0 }}><Terminal size={18} color="var(--accent-blue)" /> Developer Event Logs Console</div>
        <button className="btn btn-secondary" onClick={handleClear}>Clear Console</button>
      </div>

      <div className="terminal-console" ref={listRef} style={{ marginTop: '1rem' }}>
        {logs.length === 0 ? (
          <div className="terminal-empty">Console Ready. Waiting for Swarm Node events...</div>
        ) : (
          logs.map((log, i) => {
            const level = log.level.toUpperCase();
            let badgeClass = 'badge-info';
            if (level === 'ERROR') badgeClass = 'badge-danger';
            else if (level === 'WARNING') badgeClass = 'badge-warning';
            else if (level === 'SUCCESS') badgeClass = 'badge-success';

            return (
              <div className="terminal-line" key={i}>
                <span className="terminal-time">[{log.timestamp}]</span>
                <span className={`terminal-badge ${badgeClass}`}>{level}</span>
                <span className="terminal-source">[{log.source}]</span>
                <span className="terminal-msg">{log.message}</span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
