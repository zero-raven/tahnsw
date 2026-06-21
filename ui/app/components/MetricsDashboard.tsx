"use client";

import { useEffect, useState, useRef } from "react";
import { motion, useInView } from "framer-motion";
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, BarChart, Bar, Cell, PieChart, Pie,
    LabelList, ReferenceLine,
} from "recharts";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface CurvePoint { ef: number; recall: number; qps: number; }
interface BenchmarkData {
    hnsw: { curve: CurvePoint[]; build_time: number };
    tahnsw: { curve: CurvePoint[]; build_time: number; stats: { hub_pct: number; leaf_pct: number; normal_pct: number; mean_M_eff: number; edge_reduction_pct: number } };
}

const CUSTOM_TOOLTIP_STYLE = {
    background: "var(--surface-2)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    color: "var(--text)",
    fontSize: "0.78rem",
    fontFamily: "var(--mono)",
    padding: "8px 12px",
};

function SectionTitle({ children }: { children: React.ReactNode }) {
    return (
        <h2 style={{ fontSize: "1.25rem", fontWeight: 700, color: "var(--text)", marginBottom: 6 }}>
            {children}
        </h2>
    );
}

function StatPill({
    label, value, color, hint,
}: {
    label: string; value: string; color?: string; hint?: string;
}) {
    return (
        <div style={{
            background: "var(--surface-2)", border: "1px solid var(--border)",
            borderRadius: 12, padding: "12px 18px", display: "flex",
            flexDirection: "column", gap: 4, minWidth: 120, flex: "1 1 auto",
        }}>
            <span style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</span>
            <span style={{ fontSize: "1.15rem", fontWeight: 700, fontFamily: "var(--mono)", color: color || "var(--text)" }}>{value}</span>
            {hint && <span style={{ fontSize: "0.65rem", color: "var(--text-dim)" }}>{hint}</span>}
        </div>
    );
}

/** Custom tooltip so recharts doesn't show raw dataKey names */
function RecallQpsTooltip({ active, payload, label }: any) {
    if (!active || !payload?.length) return null;
    return (
        <div style={CUSTOM_TOOLTIP_STYLE}>
            <div style={{ marginBottom: 4, color: "var(--text-muted)" }}>
                Recall: <strong style={{ color: "var(--text)", fontFamily: "var(--mono)" }}>{Number(label).toFixed(4)}</strong>
            </div>
            {payload.map((p: any) => (
                <div key={p.dataKey} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{ width: 8, height: 8, borderRadius: "50%", background: p.color }} />
                    <span style={{ color: "var(--text-muted)" }}>{p.dataKey === "hnsw_qps" ? "HNSW" : "TAHNSW"}</span>
                    <span style={{ fontFamily: "var(--mono)", marginLeft: "auto", paddingLeft: 12, color: "var(--text)" }}>
                        {Number(p.value).toFixed(0)} QPS
                    </span>
                </div>
            ))}
        </div>
    );
}

function BuildTooltip({ active, payload }: any) {
    if (!active || !payload?.length) return null;
    return (
        <div style={CUSTOM_TOOLTIP_STYLE}>
            <span style={{ color: "var(--text-muted)" }}>{payload[0].payload.name} build: </span>
            <strong style={{ fontFamily: "var(--mono)", color: "var(--text)" }}>{Number(payload[0].value).toFixed(3)}s</strong>
        </div>
    );
}

export default function MetricsDashboard() {
    const ref = useRef<HTMLDivElement>(null);
    const inView = useInView(ref, { once: true, margin: "-80px" });
    const [data, setData] = useState<BenchmarkData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    useEffect(() => {
        if (!inView || data || loading) return;
        setLoading(true);
        fetch(`${API}/api/benchmark`)
            .then((r) => r.json())
            .then((d) => { setData(d); setLoading(false); })
            .catch(() => { setError("Could not load benchmark data."); setLoading(false); });
    }, [inView]);

    // Merge HNSW + TAHNSW curves for the Recall-QPS chart, sorted by recall asc
    const recallQpsCurve = data
        ? data.hnsw.curve
            .map((h, i) => ({
                recall: parseFloat(h.recall.toFixed(4)),
                hnsw_qps: Math.round(h.qps),
                tahnsw_qps: Math.round(data.tahnsw.curve[i]?.qps ?? 0),
            }))
            .sort((a, b) => a.recall - b.recall)
        : [];

    const buildBarData = data
        ? [
            { name: "HNSW", time: data.hnsw.build_time },
            { name: "TAHNSW", time: data.tahnsw.build_time },
        ]
        : [];

    const pieData = data
        ? [
            { name: "Hub", value: data.tahnsw.stats.hub_pct, fill: "#6c63ff" },
            { name: "Leaf", value: data.tahnsw.stats.leaf_pct, fill: "#00d4ff" },
            { name: "Normal", value: data.tahnsw.stats.normal_pct, fill: "#5a5a82" },
        ]
        : [];

    // Determine if TAHNSW is faster or slower for the speedup badge
    const tahnsw_faster = data ? data.tahnsw.build_time < data.hnsw.build_time : false;
    const speedupLabel = data
        ? tahnsw_faster
            ? `${(100 * (1 - data.tahnsw.build_time / data.hnsw.build_time)).toFixed(1)}% faster`
            : `${(100 * (data.tahnsw.build_time / data.hnsw.build_time - 1)).toFixed(1)}% slower`
        : "";

    return (
        <div ref={ref} style={{ width: "100%", maxWidth: 1100 }}>
            <motion.div
                initial={{ opacity: 0, y: 32 }}
                animate={inView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.5 }}
            >
                {/* Section header */}
                <div style={{ textAlign: "center", marginBottom: 40 }}>
                    <div style={{ display: "inline-flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                        <span style={{
                            display: "inline-block",
                            width: 8, height: 8, borderRadius: "50%",
                            background: "var(--primary)",
                            boxShadow: "0 0 8px var(--primary-glow)",
                        }} />
                        <span style={{
                            fontSize: "0.7rem", fontWeight: 600, letterSpacing: "0.1em",
                            textTransform: "uppercase", color: "var(--primary)",
                        }}>
                            Performance Metrics
                        </span>
                    </div>
                    <SectionTitle>Benchmark Results</SectionTitle>
                    <p style={{ color: "var(--text-muted)", fontSize: "0.88rem" }}>
                        {data
                            ? `${data.hnsw.curve.length}-point ef sweep · 200 queries · cosine distance`
                            : "Running against movie corpus · cosine distance"}
                    </p>
                </div>

                {loading && (
                    <div style={{ display: "flex", alignItems: "center", gap: 10, justifyContent: "center", padding: "48px 0", color: "var(--text-muted)", fontSize: "0.88rem" }}>
                        <div className="pulse-dot" />
                        Running benchmark sweep…
                    </div>
                )}

                {error && <p style={{ color: "var(--warning)", textAlign: "center" }}>{error}</p>}

                {data && (
                    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

                        {/* ── Row 1: Recall-QPS + Build Time ─────────────────── */}
                        <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>

                            {/* Recall-QPS line chart */}
                            <div className="card" style={{ flex: "2 1 380px", padding: "20px 24px 16px" }}>
                                <div style={{ marginBottom: 16 }}>
                                    <h3 style={{ fontSize: "0.92rem", fontWeight: 700, color: "var(--text)" }}>
                                        Recall vs. Queries-per-Second
                                    </h3>
                                    <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 3 }}>
                                        Curves swept across ef values — higher &amp; rightward is better
                                    </p>
                                </div>

                                <ResponsiveContainer width="100%" height={260}>
                                    <LineChart
                                        data={recallQpsCurve}
                                        margin={{ top: 8, right: 16, bottom: 32, left: 16 }}
                                    >
                                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                                        <XAxis
                                            dataKey="recall"
                                            type="number"
                                            domain={["dataMin", "dataMax"]}
                                            tickFormatter={(v) => v.toFixed(3)}
                                            tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                                            label={{
                                                value: "Recall @ k",
                                                position: "insideBottom",
                                                offset: -18,
                                                fill: "var(--text-muted)",
                                                fontSize: 11,
                                            }}
                                            tickCount={6}
                                        />
                                        <YAxis
                                            tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                                            tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : String(v)}
                                            label={{
                                                value: "QPS",
                                                angle: -90,
                                                position: "insideLeft",
                                                offset: -4,
                                                fill: "var(--text-muted)",
                                                fontSize: 11,
                                            }}
                                            width={52}
                                        />
                                        <ReferenceLine
                                            x={1}
                                            stroke="var(--border)"
                                            strokeDasharray="4 2"
                                            label={{ value: "Perfect recall", position: "top", fill: "var(--text-dim)", fontSize: 10 }}
                                        />
                                        <Tooltip content={<RecallQpsTooltip />} />
                                        <Line
                                            type="monotone"
                                            dataKey="hnsw_qps"
                                            stroke="var(--hnsw)"
                                            strokeWidth={2}
                                            dot={false}
                                            activeDot={{ r: 4, fill: "var(--hnsw)", strokeWidth: 0 }}
                                            name="hnsw_qps"
                                        />
                                        <Line
                                            type="monotone"
                                            dataKey="tahnsw_qps"
                                            stroke="var(--primary)"
                                            strokeWidth={3}
                                            dot={false}
                                            activeDot={{ r: 5, fill: "var(--primary)", strokeWidth: 0 }}
                                            name="tahnsw_qps"
                                        />
                                    </LineChart>
                                </ResponsiveContainer>

                                {/* Legend */}
                                <div style={{ display: "flex", gap: 20, marginTop: 8, paddingLeft: 4 }}>
                                    {[
                                        { color: "var(--hnsw)", label: "HNSW", dash: false },
                                        { color: "var(--primary)", label: "TAHNSW (ours)", dash: false },
                                    ].map((l) => (
                                        <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: "0.78rem", color: "var(--text-muted)" }}>
                                            <div style={{ width: 22, height: 2.5, background: l.color, borderRadius: 2 }} />
                                            {l.label}
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Build time bar chart */}
                            <div className="card" style={{ flex: "1 1 220px", padding: "20px 24px 16px" }}>
                                <div style={{ marginBottom: 16 }}>
                                    <h3 style={{ fontSize: "0.92rem", fontWeight: 700, color: "var(--text)" }}>
                                        Index Build Time
                                    </h3>
                                    <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 3 }}>
                                        Wall-clock seconds to build on corpus
                                    </p>
                                </div>

                                <ResponsiveContainer width="100%" height={200}>
                                    <BarChart
                                        data={buildBarData}
                                        layout="vertical"
                                        margin={{ left: 0, right: 52, top: 4, bottom: 4 }}
                                    >
                                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                                        <XAxis
                                            type="number"
                                            tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                                            tickFormatter={(v) => `${v.toFixed(1)}s`}
                                        />
                                        <YAxis
                                            dataKey="name"
                                            type="category"
                                            tick={{ fill: "var(--text)", fontSize: 13, fontWeight: 600 }}
                                            width={68}
                                        />
                                        <Tooltip content={<BuildTooltip />} />
                                        <Bar dataKey="time" radius={[0, 6, 6, 0]} maxBarSize={36}>
                                            {buildBarData.map((entry) => (
                                                <Cell
                                                    key={entry.name}
                                                    fill={entry.name === "HNSW" ? "var(--hnsw)" : "var(--primary)"}
                                                />
                                            ))}
                                            <LabelList
                                                dataKey="time"
                                                position="right"
                                                formatter={(v: number) => `${v.toFixed(2)}s`}
                                                style={{ fill: "var(--text-muted)", fontSize: 11, fontFamily: "var(--mono)" }}
                                            />
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>

                                {/* Speedup badge — only shown when TAHNSW is faster */}
                                {tahnsw_faster && (
                                    <div style={{
                                        marginTop: 14, padding: "10px 14px",
                                        background: "var(--surface-2)", borderRadius: 10,
                                        display: "flex", alignItems: "center", justifyContent: "space-between",
                                    }}>
                                        <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>TAHNSW vs HNSW</span>
                                        <span style={{
                                            fontSize: "0.9rem", fontWeight: 700,
                                            fontFamily: "var(--mono)",
                                            color: "var(--success)",
                                        }}>
                                            ▼ {speedupLabel}
                                        </span>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* ── Row 2: Donut + Stats ────────────────────────────── */}
                        <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>

                            {/* Node classification donut */}
                            <div className="card" style={{ flex: "1 1 280px", padding: "20px 24px" }}>
                                <div style={{ marginBottom: 14 }}>
                                    <h3 style={{ fontSize: "0.92rem", fontWeight: 700, color: "var(--text)" }}>
                                        TAHNSW Node Classification
                                    </h3>
                                    <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 3 }}>
                                        Based on clustering coefficient C(v)
                                    </p>
                                </div>

                                <div style={{ display: "flex", alignItems: "center", gap: 24, flexWrap: "wrap" }}>
                                    <div style={{ width: 180, height: 180, flexShrink: 0 }}>
                                        <ResponsiveContainer width="100%" height="100%">
                                            <PieChart>
                                                <Pie
                                                    data={pieData}
                                                    cx="50%"
                                                    cy="50%"
                                                    innerRadius={52}
                                                    outerRadius={80}
                                                    paddingAngle={3}
                                                    dataKey="value"
                                                    animationBegin={300}
                                                    animationDuration={900}
                                                >
                                                    {pieData.map((entry, index) => (
                                                        <Cell key={index} fill={entry.fill} stroke="var(--surface)" strokeWidth={2} />
                                                    ))}
                                                </Pie>
                                                <Tooltip
                                                    contentStyle={CUSTOM_TOOLTIP_STYLE}
                                                    formatter={(v: number, name: string) => [`${v.toFixed(1)}%`, name]}
                                                />
                                            </PieChart>
                                        </ResponsiveContainer>
                                    </div>

                                    <div style={{ display: "flex", flexDirection: "column", gap: 10, flex: 1, minWidth: 130 }}>
                                        {pieData.map((d) => (
                                            <div key={d.name} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                                <div style={{
                                                    width: 10, height: 10, borderRadius: "50%",
                                                    background: d.fill, flexShrink: 0,
                                                    boxShadow: `0 0 6px ${d.fill}60`,
                                                }} />
                                                <div style={{ flex: 1 }}>
                                                    <div style={{ fontSize: "0.82rem", color: "var(--text)" }}>{d.name}</div>
                                                    <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginTop: 1 }}>
                                                        {d.name === "Hub" ? "Low C(v) < 0.20" :
                                                            d.name === "Leaf" ? "High C(v) > 0.70" : "Intermediate C(v)"}
                                                    </div>
                                                </div>
                                                <span style={{
                                                    fontSize: "0.9rem", fontWeight: 700,
                                                    fontFamily: "var(--mono)", color: "var(--text)",
                                                }}>
                                                    {d.value.toFixed(1)}%
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            {/* Key stats */}
                            <div className="card" style={{ flex: "1 1 280px", padding: "20px 24px" }}>
                                <div style={{ marginBottom: 16 }}>
                                    <h3 style={{ fontSize: "0.92rem", fontWeight: 700, color: "var(--text)" }}>
                                        TAHNSW Key Stats
                                    </h3>
                                    <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 3 }}>
                                        Topology-aware construction metrics
                                    </p>
                                </div>
                                <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                                    <StatPill
                                        label="Mean M_eff"
                                        value={data.tahnsw.stats.mean_M_eff.toFixed(2)}
                                        color="var(--primary)"
                                        hint="effective connections/node"
                                    />
                                    <StatPill
                                        label="Edge Reduction"
                                        value={`${data.tahnsw.stats.edge_reduction_pct.toFixed(1)}%`}
                                        color="var(--success)"
                                        hint="vs fixed M=16 HNSW"
                                    />
                                    <StatPill
                                        label="Hub Nodes"
                                        value={`${data.tahnsw.stats.hub_pct.toFixed(1)}%`}
                                        color="var(--accent)"
                                        hint="promoted to upper layers"
                                    />
                                    <StatPill
                                        label="Leaf Nodes"
                                        value={`${data.tahnsw.stats.leaf_pct.toFixed(1)}%`}
                                        color="var(--primary)"
                                        hint="layer 0 only"
                                    />
                                    <StatPill
                                        label="TAHNSW Build"
                                        value={`${data.tahnsw.build_time.toFixed(2)}s`}
                                        hint="wall-clock"
                                    />
                                    <StatPill
                                        label="HNSW Build"
                                        value={`${data.hnsw.build_time.toFixed(2)}s`}
                                        hint="wall-clock"
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </motion.div>
        </div>
    );
}
