import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageSquare,
  Send,
  Loader2,
  AlertTriangle,
  ChevronRight,
  ChevronDown,
  Sparkles,
  Database,
} from "lucide-react";
import { PageHeader } from "@/product-ui/components/layout/page-header";
import { Card, CardContent } from "@/product-ui/components/ui/card";
import { Button } from "@/product-ui/components/ui/button";
import { Textarea } from "@/product-ui/components/ui/textarea";
import { Badge } from "@/product-ui/components/ui/badge";
import { useExecuteMode2, useModels } from "@/product-ui/api/hooks";
import type { Mode2ResponseDTO } from "@/product-ui/api/client";
import { cn } from "@/product-ui/lib/utils";

const exampleQuestions = [
  "What would happen if we increase prices by 15%?",
  "Should we expand into the European market?",
  "What's the impact of reducing customer support staff?",
  "How will supply chain disruptions affect delivery times?",
];

interface Message {
  id: string;
  type: "user" | "recommendation" | "escalation" | "error";
  content: string;
  data?: Mode2ResponseDTO;
  timestamp: Date;
}

export function DecisionSupportPage() {
  const [question, setQuestion] = useState("");
  const [domainHint, setDomainHint] = useState("");
  const [showDomainHint, setShowDomainHint] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);

  const executeMutation = useExecuteMode2();
  const { data: modelsData } = useModels();
  const domains = (Array.isArray(modelsData) ? modelsData : []).map((m) => m.domain);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-grow textarea
  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setQuestion(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  };

  const handleAsk = () => {
    if (!question.trim()) return;
    const userMsg: Message = { id: `user-${Date.now()}`, type: "user", content: question, timestamp: new Date() };
    setMessages((prev) => [...prev, userMsg]);
    const q = question;
    setQuestion("");

    executeMutation.mutate(
      { query: q, domainHint: domainHint || undefined },
      {
        onSuccess: (data) => {
          if (data.escalate_to_mode1) {
            setMessages((prev) => [...prev, { id: `esc-${Date.now()}`, type: "escalation", content: data.escalation_reason ?? "Escalation required.", data, timestamp: new Date() }]);
          } else {
            setMessages((prev) => [...prev, { id: `rec-${Date.now()}`, type: "recommendation", content: data.recommendation ?? "No recommendation provided.", data, timestamp: new Date() }]);
          }
        },
        onError: () => {
          setMessages((prev) => [...prev, { id: `err-${Date.now()}`, type: "error", content: "Failed to process your question. Is the API running?", timestamp: new Date() }]);
        },
      }
    );
  };

  return (
    <div className="flex flex-col h-[calc(100vh-160px)] lg:h-[calc(100vh-140px)]">
      <PageHeader title="Decision Support" description="Ask questions and get evidence-backed causal recommendations." />

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto pb-6 space-y-5">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full">
            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="text-center max-w-lg px-4">
              <div className="flex items-center justify-center w-20 h-20 rounded-3xl bg-gradient-to-br from-[var(--color-accent-500)] to-[var(--color-accent-700)] shadow-[var(--shadow-glow)] mx-auto mb-8">
                <MessageSquare className="w-9 h-9 text-white" />
              </div>
              <h2 className="text-[22px] font-bold text-[var(--text-primary)] mb-3">Ask a Decision Question</h2>
              <p className="text-[15px] text-[var(--text-secondary)] mb-8 leading-relaxed">
                Your question will be analyzed using causal world models and evidence from your documents.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {exampleQuestions.map((q) => (
                  <button
                    key={q}
                    onClick={() => setQuestion(q)}
                    className="text-left p-4 rounded-2xl border border-[var(--border-primary)] hover:border-[var(--color-accent-400)]/40 hover:bg-[var(--color-accent-500)]/5 text-[13px] text-[var(--text-secondary)] transition-all duration-200 hover:shadow-[var(--shadow-sm)]"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </motion.div>
          </div>
        ) : (
          <AnimatePresence>
            {messages.map((msg) => (
              <motion.div key={msg.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className={cn("max-w-3xl", msg.type === "user" ? "ml-auto" : "mr-auto")}>
                {msg.type === "user" ? (
                  <div className="bg-gradient-to-br from-[var(--color-accent-500)] to-[var(--color-accent-600)] text-white rounded-2xl rounded-br-lg px-5 py-3.5 shadow-[var(--shadow-button)]">
                    <p className="text-[14px] leading-relaxed">{msg.content}</p>
                  </div>
                ) : msg.type === "error" ? (
                  <Card className="border-red-500/20 bg-red-500/5">
                    <CardContent className="p-5 flex items-start gap-3">
                      <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
                      <p className="text-[14px] text-[var(--text-secondary)]">{msg.content}</p>
                    </CardContent>
                  </Card>
                ) : msg.type === "escalation" ? (
                  <EscalationCard data={msg.data as Mode2ResponseDTO} />
                ) : (
                  <RecommendationCard data={msg.data as Mode2ResponseDTO} />
                )}
              </motion.div>
            ))}
            {executeMutation.isPending && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-3xl">
                <Card className="border-[var(--color-accent-500)]/10">
                  <CardContent className="p-5 flex items-center gap-4">
                    <Loader2 className="w-5 h-5 text-[var(--color-accent-500)] animate-spin" />
                    <div>
                      <p className="text-[14px] font-semibold text-[var(--text-primary)]">Analyzing your question...</p>
                      <p className="text-[12px] text-[var(--text-tertiary)]">Retrieving evidence and tracing causal paths</p>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Bar */}
      <div className="shrink-0 border-t border-[var(--border-primary)] pt-5 bg-[var(--bg-base)]">
        {showDomainHint && domains.length > 0 && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} className="mb-4">
            <div className="flex items-center gap-3 flex-wrap">
              <span className="text-[12px] font-semibold text-[var(--text-secondary)]">Domain:</span>
              {domains.map((d) => (
                <button key={d} onClick={() => setDomainHint(d)} className={cn("px-3 py-1.5 rounded-full text-[12px] font-medium transition-all", domainHint === d ? "bg-[var(--color-accent-500)] text-white shadow-sm" : "bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-[var(--bg-input)]")}>{d}</button>
              ))}
              {domainHint && <button onClick={() => setDomainHint("")} className="text-[12px] text-[var(--text-tertiary)] hover:text-[var(--text-primary)] font-medium">Clear</button>}
            </div>
          </motion.div>
        )}
        <div className="flex items-end gap-3">
          <Button variant="ghost" size="icon" onClick={() => setShowDomainHint(!showDomainHint)} className={cn("shrink-0", showDomainHint && "text-[var(--color-accent-500)]")}>
            <Database className="w-4 h-4" />
          </Button>
          <div className="flex-1">
            <Textarea
              ref={textareaRef}
              value={question}
              onChange={handleTextareaChange}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleAsk(); } }}
              placeholder="Ask a decision question..."
              rows={1}
              className="min-h-[44px] max-h-[120px] py-3 resize-none rounded-2xl"
            />
          </div>
          <Button onClick={handleAsk} disabled={!question.trim() || executeMutation.isPending} size="icon" aria-label="Send message" className="shrink-0 h-[44px] w-[44px] rounded-2xl">
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}

/* ─── Confidence helpers ─── */
const CONF_META: Record<string, { label: string; pct: number; variant: "success" | "warning" | "error" }> = {
  high:   { label: "High",   pct: 90, variant: "success" },
  medium: { label: "Medium", pct: 60, variant: "warning" },
  low:    { label: "Low",    pct: 30, variant: "error" },
};

/* ─── Recommendation Card ─── */
function RecommendationCard({ data }: { data: Mode2ResponseDTO }) {
  const [expanded, setExpanded] = useState(false);
  const conf = data.confidence ? CONF_META[data.confidence.toLowerCase()] ?? null : null;
  const paths = data.causal_paths ?? [];
  const risks = data.risks ?? [];
  const actions = data.suggested_actions ?? [];
  const auditLog = data.audit_log ?? [];

  return (
    <Card className="border-[var(--color-accent-500)]/15 shadow-[var(--shadow-lg)]">
      <CardContent className="p-6 space-y-4">
        {/* ── Header: icon + recommendation + confidence ── */}
        <div className="flex items-start gap-4">
          <div className="flex items-center justify-center w-10 h-10 rounded-2xl bg-[var(--color-accent-500)]/10 shrink-0">
            <Sparkles className="w-5 h-5 text-[var(--color-accent-500)]" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <p className="text-[11px] font-bold text-[var(--text-secondary)] uppercase tracking-widest">Recommendation</p>
              {conf && (
                <Badge variant={conf.variant}>
                  {conf.label} confidence
                </Badge>
              )}
            </div>
            <p className="text-[14px] text-[var(--text-primary)] leading-relaxed">{data.recommendation}</p>
          </div>
        </div>

        {/* ── Expected Outcome (always visible when present) ── */}
        {data.expected_outcome && (
          <div className="ml-14 p-3 rounded-xl bg-[var(--bg-tertiary)] border border-[var(--border-primary)]">
            <p className="text-[11px] font-bold text-[var(--text-tertiary)] uppercase tracking-widest mb-1">Expected Outcome</p>
            <p className="text-[13px] text-[var(--text-secondary)] leading-relaxed">{data.expected_outcome}</p>
          </div>
        )}

        {/* ── Show details toggle ── */}
        <button onClick={() => setExpanded(!expanded)} className="flex items-center gap-1.5 text-[12px] text-[var(--text-tertiary)] hover:text-[var(--text-secondary)] transition-colors font-medium">
          {expanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          {expanded ? "Show less" : "Show details"}
        </button>

        <AnimatePresence>
          {expanded && (
            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="space-y-5 overflow-hidden">

              {/* ── Metadata grid ── */}
              <div className="grid grid-cols-2 gap-4 text-[13px]">
                {data.stage && <div><span className="font-semibold text-[var(--text-tertiary)]">Stage:</span> <span className="text-[var(--text-secondary)]">{data.stage}</span></div>}
                {data.model_used && <div><span className="font-semibold text-[var(--text-tertiary)]">Model:</span> <span className="text-[var(--text-secondary)]">{data.model_used}</span></div>}
                <div><span className="font-semibold text-[var(--text-tertiary)]">Evidence:</span> <span className="text-[var(--text-secondary)]">{data.evidence_count} items</span></div>
                <div><span className="font-semibold text-[var(--text-tertiary)]">Trace ID:</span> <span className="text-[var(--text-secondary)] font-mono text-[11px]">{data.trace_id}</span></div>
              </div>

              {/* ── Causal Paths ── */}
              {paths.length > 0 && (
                <div>
                  <p className="text-[11px] font-bold text-[var(--text-tertiary)] uppercase tracking-widest mb-2">Causal Paths Traced</p>
                  <div className="space-y-2">
                    {paths.map((cp, i) => (
                      <div key={i} className="p-3 rounded-xl bg-[var(--bg-tertiary)] border border-[var(--border-primary)]">
                        <div className="flex items-center gap-1.5 mb-1.5 flex-wrap">
                          {cp.path.map((node, j) => (
                            <span key={j} className="flex items-center gap-1.5">
                              <span className="px-2 py-0.5 rounded-md bg-[var(--color-accent-500)]/10 text-[var(--color-accent-500)] text-[11px] font-semibold">
                                {node.replace(/_/g, " ")}
                              </span>
                              {j < cp.path.length - 1 && <ChevronRight className="w-3 h-3 text-[var(--text-tertiary)]" />}
                            </span>
                          ))}
                          <Badge variant={cp.strength === "strong" ? "success" : cp.strength === "moderate" ? "warning" : "error"} className="ml-auto text-[10px]">{cp.strength}</Badge>
                        </div>
                        <p className="text-[12px] text-[var(--text-secondary)] leading-relaxed">{cp.mechanism_chain}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* ── Reasoning Trace ── */}
              {data.reasoning_trace && (
                <div>
                  <p className="text-[11px] font-bold text-[var(--text-tertiary)] uppercase tracking-widest mb-2">Reasoning Trace</p>
                  <div className="p-3 rounded-xl bg-[var(--bg-tertiary)] border border-[var(--border-primary)]">
                    <p className="text-[12px] text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap">{data.reasoning_trace}</p>
                  </div>
                </div>
              )}

              {/* ── Risks ── */}
              {risks.length > 0 && (
                <div>
                  <p className="text-[11px] font-bold text-[var(--text-tertiary)] uppercase tracking-widest mb-2">Risks Identified</p>
                  <ul className="space-y-1">
                    {risks.map((r, i) => (
                      <li key={i} className="flex items-start gap-2 text-[12px] text-[var(--text-secondary)]">
                        <AlertTriangle className="w-3.5 h-3.5 text-amber-500 shrink-0 mt-0.5" />
                        {r}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* ── Suggested Actions ── */}
              {actions.length > 0 && (
                <div>
                  <p className="text-[11px] font-bold text-[var(--text-tertiary)] uppercase tracking-widest mb-2">Suggested Actions</p>
                  <ul className="space-y-1">
                    {actions.map((a, i) => (
                      <li key={i} className="flex items-start gap-2 text-[12px] text-[var(--text-secondary)]">
                        <ChevronRight className="w-3.5 h-3.5 text-[var(--color-accent-500)] shrink-0 mt-0.5" />
                        {a}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* ── Audit Log ── */}
              {auditLog.length > 0 && (
                <div>
                  <p className="text-[11px] font-bold text-[var(--text-tertiary)] uppercase tracking-widest mb-2">Audit Trail</p>
                  <div className="space-y-1.5">
                    {auditLog.map((entry, i) => (
                      <div key={i} className="flex items-start gap-3 text-[11px]">
                        <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-accent-500)] shrink-0 mt-1.5" />
                        <div className="flex-1">
                          <span className="font-semibold text-[var(--text-secondary)]">{entry.action}</span>
                          {entry.data && Object.keys(entry.data).length > 0 && (
                            <span className="text-[var(--text-tertiary)] ml-2">
                              {Object.entries(entry.data).map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : String(v)}`).join(" · ")}
                            </span>
                          )}
                        </div>
                        <span className="text-[var(--text-tertiary)] font-mono shrink-0">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

            </motion.div>
          )}
        </AnimatePresence>
      </CardContent>
    </Card>
  );
}

/* ─── Escalation Card ─── */
function EscalationCard({ data }: { data: Mode2ResponseDTO }) {
  return (
    <Card className="border-amber-500/20 bg-amber-500/5">
      <CardContent className="p-6">
        <div className="flex items-start gap-4">
          <div className="flex items-center justify-center w-10 h-10 rounded-2xl bg-amber-500/10 shrink-0">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
          </div>
          <div className="flex-1 space-y-3">
            <div>
              <Badge variant="warning" className="mb-2">Escalation Required</Badge>
              <p className="text-[14px] text-[var(--text-primary)] leading-relaxed">{data.escalation_reason ?? "This query requires a Mode 1 world model build."}</p>
            </div>
            <div className="space-y-2 text-[13px] text-[var(--text-secondary)]">
              {data.stage && <div><span className="font-semibold text-[var(--text-tertiary)]">Stage:</span> {data.stage}</div>}
              {data.model_used && <div><span className="font-semibold text-[var(--text-tertiary)]">Model:</span> {data.model_used}</div>}
              <div><span className="font-semibold text-[var(--text-tertiary)]">Evidence:</span> {data.evidence_count} items</div>
              {data.error && <div><span className="font-semibold text-red-400">Error:</span> {data.error}</div>}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
