import json
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go

# Custom CSS for professional appearance
custom_css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        max-width: 1200px;
    }
    .gr-button-primary {
        background-color: #2563eb;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .gr-button-primary:hover {
        background-color: #1d4ed8;
    }
    footer {
        display: none !important;
    }
    .gr-box {
        border-radius: 8px;
        border-color: #e5e7eb;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f9fafb;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
"""


# Mock analysis function for MVP demo
def analyze_video(video_file, _scene_threshold, _include_visual_analysis):
    """Simulates video analysis - replace with actual implementation"""

    if video_file is None:
        return None, None, None, None, None, "Please upload a video first"

    # Simulate processing time
    for i in range(5):
        time.sleep(0.5)  # In real implementation, this would be actual processing
        yield (
            f"Processing... {20 * (i + 1)}%",
            None,
            None,
            None,
            None,
            f"⏳ Extracting audio... Processing scenes... Analyzing speech... ({20 * (i + 1)}%)",
        )

    # Mock results for demonstration
    transcript = """Good morning everyone. Today I'll be presenting our quarterly results.
As you can see on this slide, we've had significant growth in, um, all major markets.
Our revenue increased by 15% compared to last quarter, which is, uh, really exciting for the team.
Let me walk you through the key metrics..."""

    # Key metrics
    metrics = {
        "📊 Speaking Rate": "145 WPM",
        "⏱️ Duration": "5:23",
        "💬 Total Words": "782",
        "🔤 Filler Words": "12 (um: 7, uh: 5)",
        "😊 Sentiment": "Positive (0.72)",
        "📖 Readability": "Grade 8",
    }

    # Create visualizations
    # 1. Speaking rate over time
    speaking_rate_chart = go.Figure()
    speaking_rate_chart.add_trace(
        go.Scatter(
            x=["0:00", "1:00", "2:00", "3:00", "4:00", "5:00"],
            y=[135, 145, 150, 142, 148, 140],
            mode="lines+markers",
            name="WPM",
            line={"color": "#2563eb", "width": 3},
        )
    )
    speaking_rate_chart.add_hline(
        y=145, line_dash="dash", annotation_text="Target: 145 WPM", line_color="gray"
    )
    speaking_rate_chart.update_layout(
        title="Speaking Rate Over Time",
        xaxis_title="Time",
        yaxis_title="Words Per Minute",
        template="plotly_white",
        height=300,
    )

    # 2. Scene breakdown pie chart
    scene_chart = go.Figure(
        data=[
            go.Pie(
                labels=[
                    "Introduction",
                    "Market Analysis",
                    "Results",
                    "Future Plans",
                    "Q&A",
                ],
                values=[65, 120, 180, 95, 60],
                hole=0.3,
                marker_colors=["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"],
            )
        ]
    )
    scene_chart.update_layout(
        title="Time per Section (seconds)", template="plotly_white", height=300
    )

    # Generate sample report
    report_data = {
        "metadata": {
            "filename": Path(video_file.name).name,
            "analyzed_at": datetime.now().isoformat(),
            "duration": "5:23",
        },
        "metrics": metrics,
        "transcript": transcript,
        "analysis": {
            "strengths": [
                "Clear structure with defined sections",
                "Good pacing overall",
                "Positive and engaging tone",
            ],
            "improvements": [
                "Reduce filler words (12 occurrences)",
                "Add more pauses between sections",
                "Increase volume consistency",
            ],
        },
    }

    # Save report
    report_path = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    status = "✅ Analysis complete! Your video has been processed successfully."

    yield transcript, metrics, speaking_rate_chart, scene_chart, report_path, status


# Create Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(), css=custom_css, title="FrameFocus - Video Analysis"
) as demo:
    # Header
    gr.Markdown(
        """
        <div class="main-header">
            <h1>🎥 FrameFocus</h1>
            <p style="color: #6b7280; font-size: 1.1em;">Professional Video Presentation Analysis</p>
        </div>
        """
    )

    with gr.Row():
        # Left column - Input
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Video")
            video_input = gr.Video(label="Select your video file", height=300)

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                scene_threshold = gr.Slider(
                    minimum=0.2,
                    maximum=0.8,
                    value=0.4,
                    step=0.1,
                    label="Scene Detection Sensitivity",
                    info="Lower = more scenes detected",
                )
                include_visual = gr.Checkbox(
                    label="Include Visual Analysis",
                    value=True,
                    info="Analyze slides and visual content",
                )

            analyze_btn = gr.Button("🚀 Start Analysis", variant="primary", size="lg")

            status_text = gr.Textbox(
                label="Status",
                value="Ready to analyze your video",
                interactive=False,
                lines=2,
            )

        # Right column - Results
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Analysis Results")

            # Metrics cards
            metrics_output = gr.JSON(label="Key Metrics", elem_classes=["metric-card"])

            # Charts
            with gr.Row():
                rate_chart = gr.Plot(label="Speaking Rate Analysis")
                scene_chart = gr.Plot(label="Scene Breakdown")

            # Transcript
            transcript_output = gr.Textbox(
                label="📝 Transcript Preview", lines=6, max_lines=10, interactive=False
            )

            # Download
            report_file = gr.File(label="📥 Download Full Report", visible=True)

    # Footer info
    gr.Markdown(
        """
        ---
        <div style="text-align: center; color: #9ca3af; margin-top: 2rem;">
            <p>FrameFocus MVP v1.0 | Process videos locally for complete privacy</p>
            <p style="font-size: 0.9em;">Supports MP4, MOV, AVI • Max 500MB • Processing time ~30s per minute of video</p>
        </div>
        """
    )

    # Wire up the analysis
    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input, scene_threshold, include_visual],
        outputs=[
            transcript_output,
            metrics_output,
            rate_chart,
            scene_chart,
            report_file,
            status_text,
        ],
    )

    # Example videos section
    with gr.Accordion("📚 Example Videos", open=False):
        gr.Examples(
            examples=[
                ["examples/presentation_good.mp4"],
                ["examples/presentation_needs_work.mp4"],
            ],
            inputs=video_input,
            label="Try these sample videos",
        )

# Launch instructions
if __name__ == "__main__":
    # For local development
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Set to True for temporary public URL
        inbrowser=True,
    )

    # For production deployment options:
    # 1. Local only: demo.launch(share=False)
    # 2. Temporary public URL: demo.launch(share=True)
    # 3. Deploy to Hugging Face Spaces (free hosting)
    # 4. Deploy to your own server with: demo.launch(server_name="0.0.0.0")
