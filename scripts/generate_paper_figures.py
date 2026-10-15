"""
generate_paper_figures.py
Generates publication-quality 300 DPI figures for the Smart Campus Parking Paper.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs("figures", exist_ok=True)

# Set high-quality IEEE/ACM aesthetic styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# ==============================================================================
# FIGURE 1: Three-Tier System Architecture
# ==============================================================================
def make_fig1_architecture():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Background tier panels
    # Tier 1: Perception
    t1 = patches.FancyBboxPatch((3, 68), 94, 28, boxstyle="round,pad=1.5",
                               ec="#1E3A8A", fc="#EFF6FF", lw=1.5)
    ax.add_patch(t1)
    ax.text(6, 92, "PERCEPTION TIER: EDGE SENSING & IOT CAPTURE", fontsize=11, fontweight='bold', color="#1E3A8A")

    # Tier 1 Components
    c1 = patches.FancyBboxPatch((8, 71), 40, 17, boxstyle="round,pad=1", ec="#3B82F6", fc="#FFFFFF", lw=1.2)
    ax.add_patch(c1)
    ax.text(28, 83, "Zone P1 Node (ESP32-CAM + OV2640)", fontsize=9, fontweight='bold', ha='center', color="#1F2937")
    ax.text(28, 79, "20 Two-Wheeler Slots (Bikes)", fontsize=8, ha='center', color="#4B5563")
    ax.text(28, 75, "Edge Analysis / 320x240 QVGA / MJPEG", fontsize=7.5, ha='center', color="#2563EB")

    c2 = patches.FancyBboxPatch((52, 71), 40, 17, boxstyle="round,pad=1", ec="#3B82F6", fc="#FFFFFF", lw=1.2)
    ax.add_patch(c2)
    ax.text(72, 83, "Zone P3 Node (High-Res Mobile Stream)", fontsize=9, fontweight='bold', ha='center', color="#1F2937")
    ax.text(72, 79, "15 Four-Wheeler Slots (Faculty/Cars)", fontsize=8, ha='center', color="#4B5563")
    ax.text(72, 75, "RTSP/HTTP MJPEG Stream (640x480 VGA)", fontsize=7.5, ha='center', color="#2563EB")

    # Tier 2: Application Engine
    t2 = patches.FancyBboxPatch((3, 33), 94, 31, boxstyle="round,pad=1.5",
                               ec="#065F46", fc="#ECFDF5", lw=1.5)
    ax.add_patch(t2)
    ax.text(6, 60.5, "CORE APPLICATION & PROCESSING TIER (FastAPI Server)", fontsize=11, fontweight='bold', color="#065F46")

    # Tier 2 Submodules
    m1 = patches.FancyBboxPatch((6, 36), 26, 21, boxstyle="round,pad=0.8", ec="#10B981", fc="#FFFFFF", lw=1)
    ax.add_patch(m1)
    ax.text(19, 52.5, "Vision Pipeline", fontsize=8.5, fontweight='bold', ha='center', color="#065F46")
    ax.text(19, 48, "- OpenCV Thread Pool\n- INT8 CNN & Texture\n- Auto-Calibration Grid\n- Camera Failover Engine",
            fontsize=7, ha='center', va='center', color="#374151")

    m2 = patches.FancyBboxPatch((37, 36), 26, 21, boxstyle="round,pad=0.8", ec="#10B981", fc="#FFFFFF", lw=1)
    ax.add_patch(m2)
    ax.text(50, 52.5, "Allocation & Routing", fontsize=8.5, fontweight='bold', ha='center', color="#065F46")
    ax.text(50, 48, "- Role & Vehicle Filter\n- Dijkstra Shortest Path\n- Atomic Row Locks\n- Idempotency Hash (SHA)",
            fontsize=7, ha='center', va='center', color="#374151")

    m3 = patches.FancyBboxPatch((68, 36), 26, 21, boxstyle="round,pad=0.8", ec="#10B981", fc="#FFFFFF", lw=1)
    ax.add_patch(m3)
    ax.text(81, 52.5, "State & Persistence", fontsize=8.5, fontweight='bold', ha='center', color="#065F46")
    ax.text(81, 48, "- SQLite WAL (ACID)\n- 5-Min TTL Hold Manager\n- Auto-Release Daemon\n- WebSocket Broadcast",
            fontsize=7, ha='center', va='center', color="#374151")

    # Tier 3: Presentation
    t3 = patches.FancyBboxPatch((3, 2), 94, 27, boxstyle="round,pad=1.5",
                               ec="#7C2D12", fc="#FFF7ED", lw=1.5)
    ax.add_patch(t3)
    ax.text(6, 25.5, "PRESENTATION & CLIENT INTERACTION TIER", fontsize=11, fontweight='bold', color="#7C2D12")

    u1 = patches.FancyBboxPatch((8, 5), 40, 17, boxstyle="round,pad=1", ec="#F97316", fc="#FFFFFF", lw=1.2)
    ax.add_patch(u1)
    ax.text(28, 17, "Customer / Driver Portal", fontsize=9, fontweight='bold', ha='center', color="#1F2937")
    ax.text(28, 13, "1-Tap Role-Based Smart Reservation", fontsize=8, ha='center', color="#4B5563")
    ax.text(28, 9, "Turn-by-Turn GPS + Google Maps Nav", fontsize=7.5, ha='center', color="#EA580C")

    u2 = patches.FancyBboxPatch((52, 5), 40, 17, boxstyle="round,pad=1", ec="#F97316", fc="#FFFFFF", lw=1.2)
    ax.add_patch(u2)
    ax.text(72, 17, "Admin & Operations Portal", fontsize=9, fontweight='bold', ha='center', color="#1F2937")
    ax.text(72, 13, "Live 35-Slot Multi-Zone Matrix", fontsize=8, ha='center', color="#4B5563")
    ax.text(72, 9, "ROI Stream, Manual Free, Mispark Alert", fontsize=7.5, ha='center', color="#EA580C")

    # Arrows between tiers
    # Tier 1 to Tier 2
    ax.annotate('', xy=(28, 64), xytext=(28, 71),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#2563EB"))
    ax.text(29, 67.5, "WiFi HTTP/MJPEG", fontsize=7.5, color="#1E40AF", fontweight='bold')

    ax.annotate('', xy=(72, 64), xytext=(72, 71),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#2563EB"))
    ax.text(73, 67.5, "WiFi HTTP/MJPEG", fontsize=7.5, color="#1E40AF", fontweight='bold')

    # Tier 2 to Tier 3
    ax.annotate('', xy=(28, 22), xytext=(28, 33),
                arrowprops=dict(arrowstyle="<->", lw=1.5, color="#EA580C"))
    ax.text(29, 27.5, "REST API + WebSocket", fontsize=7.5, color="#C2410C", fontweight='bold')

    ax.annotate('', xy=(72, 22), xytext=(72, 33),
                arrowprops=dict(arrowstyle="<->", lw=1.5, color="#EA580C"))
    ax.text(73, 27.5, "Live Push (<50ms)", fontsize=7.5, color="#C2410C", fontweight='bold')

    plt.tight_layout()
    plt.savefig("figures/fig1_system_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figures/fig1_system_architecture.png")

# ==============================================================================
# FIGURE 2: Computer Vision Detection Pipeline Flowchart
# ==============================================================================
def make_fig2_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=300)
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 42)
    ax.axis('off')

    stages = [
        ("Raw Video\nFrame Ingest\n(VGA/QVGA)", 3, 10, 16, 22, "#3B82F6"),
        ("Normalised\nPolygon ROI\nExtraction", 22, 10, 16, 22, "#10B981"),
        ("Boundary Inset\n& Grayscale\nCLAHE Norm.", 41, 10, 16, 22, "#F59E0B"),
        ("Parallel Inference\n1. INT8 TFLite CNN\n2. Texture Std-Dev", 60, 10, 17, 22, "#8B5CF6"),
        ("Temporal Filter\n3-Frame Majority\nVoting Filter", 80, 10, 14, 22, "#EC4899"),
        ("Atomic DB\nState Broadcast\n(FREE/OCCUPIED)", 97, 10, 12, 22, "#EF4444"),
    ]

    for title, x, y, w, h, col in stages:
        p = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8",
                                   ec=col, fc="#FFFFFF", lw=1.5)
        ax.add_patch(p)
        header = patches.Rectangle((x, y+h-5), w, 5, ec=col, fc=col)
        ax.add_patch(header)
        ax.text(x+w/2, y+h-2.5, "STAGE", fontsize=6.5, fontweight='bold', color="#FFFFFF", ha='center', va='center')
        ax.text(x+w/2, y+(h-5)/2, title, fontsize=7.5, fontweight='bold', color="#1F2937", ha='center', va='center')

    for i in range(len(stages)-1):
        x_start = stages[i][1] + stages[i][3]
        x_end = stages[i+1][1]
        y_mid = 21
        ax.annotate('', xy=(x_end, y_mid), xytext=(x_start, y_mid),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#4B5563", mutation_scale=12))

    # Details underneath
    ax.text(11, 5, "HTTP/MJPEG Grabber\nThread (Lock-buffered)", fontsize=6.5, ha='center', color="#4B5563")
    ax.text(30, 5, "Normalized [[x,y]x4]\nResolution-invariant", fontsize=6.5, ha='center', color="#4B5563")
    ax.text(49, 5, "12% Inset to omit\npainted white lines", fontsize=6.5, ha='center', color="#4B5563")
    ax.text(68.5, 5, "33.7 KB INT8 CNN\nstd_dev < 28.0 threshold", fontsize=6.5, ha='center', color="#4B5563")
    ax.text(87, 5, "Suppresses passing\nshadows / hand noise", fontsize=6.5, ha='center', color="#4B5563")
    ax.text(103, 5, "WebSocket push\n<50ms to portal", fontsize=6.5, ha='center', color="#4B5563")

    plt.tight_layout()
    plt.savefig("figures/fig2_detection_pipeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figures/fig2_detection_pipeline.png")

# ==============================================================================
# FIGURE 3: CNN Architecture Diagram
# ==============================================================================
def make_fig3_cnn():
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')

    layers = [
        ("Input", "64x64x1\nGrayscale", 4, 15, 9, 20, "#9CA3AF"),
        ("Conv2D #1", "8 filters 3x3\n+ BN + Pool\nOut: 31x31x8", 16, 15, 12, 20, "#3B82F6"),
        ("Conv2D #2", "16 filters 3x3\n+ BN + Pool\nOut: 14x14x16", 31, 15, 13, 20, "#2563EB"),
        ("Conv2D #3", "32 filters 3x3\n+ BN + Pool\nOut: 6x6x32", 47, 15, 13, 20, "#1D4ED8"),
        ("Flatten", "Vector:\n1,152 units\nDropout 0.3", 63, 15, 10, 20, "#F59E0B"),
        ("Dense #1", "16 units\nReLU act.", 76, 15, 9, 20, "#10B981"),
        ("Dense #2", "1 unit\nSigmoid\n[0.0, 1.0]", 88, 15, 8, 20, "#EF4444"),
    ]

    for name, desc, x, y, w, h, col in layers:
        p = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                   ec=col, fc="#F9FAFB", lw=1.5)
        ax.add_patch(p)
        top = patches.Rectangle((x, y+h-4), w, 4, ec=col, fc=col)
        ax.add_patch(top)
        ax.text(x+w/2, y+h-2, name, fontsize=7, fontweight='bold', color="#FFFFFF", ha='center', va='center')
        ax.text(x+w/2, y+(h-4)/2, desc, fontsize=6.5, color="#1F2937", ha='center', va='center')

    for i in range(len(layers)-1):
        x_s = layers[i][2] + layers[i][4]
        x_e = layers[i+1][2]
        ax.annotate('', xy=(x_e, 25), xytext=(x_s, 25),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#4B5563", mutation_scale=10))

    # Summary box at bottom
    sum_box = patches.FancyBboxPatch((5, 2), 90, 8, boxstyle="round,pad=0.5",
                                     ec="#4B5563", fc="#F3F4F6", lw=1)
    ax.add_patch(sum_box)
    ax.text(50, 6, "Total Parameters: 24,577  |  Trainable: 24,465  |  Keras FP32 Model: 365.8 KB  |  Quantized INT8 TFLite: 33.7 KB (-90.8%)",
            fontsize=7.5, fontweight='bold', ha='center', va='center', color="#111827")

    plt.tight_layout()
    plt.savefig("figures/fig3_cnn_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figures/fig3_cnn_architecture.png")

# ==============================================================================
# FIGURE 6: End-to-End Latency Breakdown Chart
# ==============================================================================
def make_fig6_latency():
    stages = [
        'Network Frame Grab\n(WiFi Camera Stream)',
        'CV Analysis / Inference\n(All 35 Slot Patches)',
        'Database Lock &\nAtomic Commit',
        'Dijkstra Routing &\nGoogle Maps Path',
        'WebSocket Broadcast\nto Client Dashboards'
    ]
    mean_lat = [22.4, 14.8, 3.2, 0.8, 12.1]
    p95_lat = [35.1, 21.0, 5.8, 1.4, 24.3]

    x = np.arange(len(stages))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300, gridspec_kw={'width_ratios': [2.8, 1]})

    rects1 = ax1.bar(x - width/2, mean_lat, width, label='Mean Latency (ms)', color='#2563EB', edgecolor='#1E40AF')
    rects2 = ax1.bar(x + width/2, p95_lat, width, label='p95 Latency (ms)', color='#F97316', edgecolor='#C2410C')

    ax1.set_ylabel('Execution Time (milliseconds)', fontsize=9, fontweight='bold')
    ax1.set_title('Subsystem Latency Breakdown (Over 100 Allocation Trials)', fontsize=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=7.5)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 42)

    for r in rects1:
        h = r.get_height()
        ax1.annotate(f'{h:.1f}', xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f'{h:.1f}', xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # Total Turnaround Comparison
    tot_labels = ['Total Turnaround\n(Mean: 53.3 ms)', 'Total Turnaround\n(p95: 87.6 ms)']
    tot_vals = [53.3, 87.6]
    tot_colors = ['#10B981', '#EF4444']
    ax2.bar([0, 1], tot_vals, color=tot_colors, edgecolor='#1F2937', width=0.55)
    ax2.set_ylabel('Total Time (ms)', fontsize=9, fontweight='bold')
    ax2.set_title('End-to-End SLA', fontsize=10, fontweight='bold')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Mean\n53.3 ms', 'p95\n87.6 ms'], fontsize=8, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 100)
    for i, v in enumerate(tot_vals):
        ax2.annotate(f'{v:.1f} ms', xy=(i, v), xytext=(0, 2), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig("figures/fig6_latency_breakdown.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figures/fig6_latency_breakdown.png")

# ==============================================================================
# FIGURE 8: Baseline Comparison
# ==============================================================================
def make_fig8_baselines():
    models = ['Canny Edge / ROI\n(Sathyabama)', 'MobileNetV2\n(Unquantized)', 'Proposed INT8 CNN\n(Our Work)', 'Proposed Hybrid CV\n(Texture Fallback)']
    acc = [88.4, 98.7, 97.8, 96.5]
    sizes_kb = [0.0, 14200.0, 33.7, 0.0]  # in KB
    esp_infer = [28.4, 650.0, 48.2, 18.5] # in ms (MobileNetV2 estimated on PSRAM ESP32)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), dpi=300)

    # Accuracy comparison
    bars1 = ax1.bar(models, acc, color=['#9CA3AF', '#60A5FA', '#10B981', '#34D399'], edgecolor='#1F2937', width=0.55)
    ax1.set_ylabel('Classification Accuracy (%)', fontsize=9, fontweight='bold')
    ax1.set_title('Detection Accuracy Comparison', fontsize=10, fontweight='bold')
    ax1.set_ylim(80, 102)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(b.get_x() + b.get_width()/2, h),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax1.set_xticklabels(models, fontsize=7.5)

    # Edge Inference Latency comparison
    bars2 = ax2.bar(models, esp_infer, color=['#9CA3AF', '#F87171', '#3B82F6', '#60A5FA'], edgecolor='#1F2937', width=0.55)
    ax2.set_ylabel('Inference Latency on ESP32 (ms)', fontsize=9, fontweight='bold')
    ax2.set_title('Edge MCU Execution Speed (ESP32 @ 240MHz)', fontsize=10, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f'{h:.1f}ms', xy=(b.get_x() + b.get_width()/2, h),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax2.set_xticklabels(models, fontsize=7.5)

    plt.tight_layout()
    plt.savefig("figures/fig8_baseline_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figures/fig8_baseline_comparison.png")

if __name__ == '__main__':
    make_fig1_architecture()
    make_fig2_pipeline()
    make_fig3_cnn()
    make_fig6_latency()
    make_fig8_baselines()
    print("All paper figures generated successfully!")
