import React from "react";
import { Link } from "react-router-dom";

// --- Simple Line Chart Component ---
// This draws the line using SVG coordinates (no extra library needed!)
const SimpleLineChart = () => {
  const points = [
    { x: 0, y: 70 }, { x: 50, y: 69 }, { x: 100, y: 68 }, 
    { x: 150, y: 69 }, { x: 200, y: 66 }, { x: 250, y: 68 }, 
    { x: 300, y: 66 }, { x: 350, y: 65 }, { x: 400, y: 67 }, 
    { x: 450, y: 66 }, { x: 500, y: 65 } // The last point (Current)
  ];

  // Convert points to an SVG path string
  const polylinePoints = points.map(p => `${p.x},${100 - p.y}`).join(" ");

  return (
    <div className="relative h-48 w-full select-none">
      {/* Y-Axis Labels */}
      <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-xs text-gray-400">
        <span>70%</span>
        <span>65%</span>
        <span>60%</span>
        <span>55%</span>
      </div>

      {/* The Chart Area */}
      <div className="ml-8 h-full border-l border-b border-gray-200 relative">
        <svg viewBox="0 0 520 50" className="h-full w-full overflow-visible" preserveAspectRatio="none">
          {/* Grid Lines (Optional) */}
          <line x1="0" y1="12.5" x2="520" y2="12.5" stroke="#f3f4f6" strokeWidth="1" />
          <line x1="0" y1="25" x2="520" y2="25" stroke="#f3f4f6" strokeWidth="1" />
          <line x1="0" y1="37.5" x2="520" y2="37.5" stroke="#f3f4f6" strokeWidth="1" />

          {/* The Trend Line */}
          <polyline
            fill="none"
            stroke="#4b5563" // Gray color
            strokeWidth="2"
            points={polylinePoints}
          />
          
          {/* The Blue Dot at the end */}
          <circle cx="500" cy="35" r="4" fill="#3b82f6" stroke="white" strokeWidth="2" />
        </svg>

        {/* Current Label floating near the dot */}
        <div className="absolute right-0 top-[30%] bg-white px-2 py-1 text-xs font-bold text-gray-800 shadow-sm border rounded">
          Current: 65%
        </div>

        {/* X-Axis Labels */}
        <div className="absolute -bottom-6 w-full flex justify-between text-xs text-gray-400">
          <span>1</span>
          <span>7</span>
          <span>15</span>
          <span>22</span>
          <span>30 days</span>
        </div>
      </div>
    </div>
  );
};

// --- Achievement Badge Component ---
const Achievement = ({ title, icon, color }) => (
  <div className="flex items-center gap-3">
    <div className={`flex h-8 w-8 items-center justify-center rounded-full text-white ${color}`}>
      {icon}
    </div>
    <span className="text-sm font-medium text-gray-700">{title}</span>
  </div>
);

// --- Main Page Component ---
export default function Progress() {
  return (
    <main className="min-h-screen bg-gray-50 px-6 py-10 font-sans">
      <div className="mx-auto max-w-4xl">
        
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Your Progress</h1>
          <p className="mt-1 text-sm text-gray-500">Last 30 Days</p>
        </div>

        {/* 1. Obesity Risk Trend Card */}
        <div className="mb-6 rounded-xl bg-white p-6 shadow-sm border border-gray-100">
          <h2 className="text-sm font-bold uppercase tracking-wide text-gray-500 mb-6">Obesity Risk Trend</h2>
          
          <SimpleLineChart />

          <div className="mt-8 text-sm font-semibold text-emerald-600 bg-emerald-50 inline-block px-3 py-1 rounded-full">
            Improvement: -5% ↓ (Good!)
          </div>
        </div>

        {/* Bottom Grid */}
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          
          {/* 2. Weight Trend Card */}
          <div className="rounded-xl bg-white p-6 shadow-sm border border-gray-100">
            <h2 className="text-sm font-bold uppercase tracking-wide text-gray-500 mb-4">Weight Trend</h2>
            <div className="space-y-4">
              <div className="text-lg text-gray-800">
                Started: <span className="font-bold">85 kg</span> → Current: <span className="font-bold">83 kg</span>
              </div>
              <div className="h-2 w-full rounded-full bg-gray-100">
                <div className="h-full w-1/2 rounded-full bg-gray-800"></div>
              </div>
              <p className="text-sm text-gray-600">
                Lost: <span className="font-bold text-gray-900">2 kg</span> in 30 days
              </p>
            </div>
          </div>

          {/* 3. Achievements Card */}
          <div className="rounded-xl bg-white p-6 shadow-sm border border-gray-100">
            <h2 className="text-sm font-bold uppercase tracking-wide text-gray-500 mb-4">Achievements</h2>
            <div className="space-y-4">
              <Achievement title="7-Day Streak" icon="🏆" color="bg-emerald-500" />
              <Achievement title="Consistent Logger" icon="📝" color="bg-blue-500" />
              <Achievement title="Risk Reducer" icon="⭐" color="bg-orange-400" />
            </div>
          </div>

        </div>

        {/* Back Link */}
        <div className="mt-8">
           <Link to="/dashboard" className="text-sm text-gray-500 hover:underline">← Back to Dashboard</Link>
        </div>

      </div>
    </main>
  );
}