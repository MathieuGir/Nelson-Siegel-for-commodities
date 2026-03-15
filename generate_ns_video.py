"""
Generate MP4 videos of Nelson-Siegel term structure evolution.
"""
import sys
import math
import webbrowser
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from tqdm import tqdm

# Setup path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from helpers.nelson_curve_helpers import NS_estimation, NS_rate
from helpers.data_helpers import load_calibration_data


def make_ns_video(
    commo_ticker: str,
    start_date: str,
    end_date: str,
    min_maturity: int = 10,
    max_maturity: int = 500,
    exclude_short_for_fit: int = 20,
    grid_step: int = 5,
    frame_stride: int = 5,
    fps: int = 20,
    dpi: int = 100,
    output_path: str = None,
) -> Path:
    """
    Generate an MP4 video of Nelson-Siegel term structure evolution.
    
    Parameters
    ----------
    commo_ticker : str
        Commodity ticker (e.g., "CT", "KC", "SB", "CC")
    start_date : str
        Start date for data range
    end_date : str
        End date for data range
    min_maturity : int
        Minimum maturity for plotting (x-axis start)
    max_maturity : int
        Maximum maturity for plotting (x-axis end)
    exclude_short_for_fit : int
        Exclude contracts with maturity < this value from NS estimation
    grid_step : int
        Step size for NS evaluation grid
    frame_stride : int
        Render every Nth date (for performance)
    fps : int
        Frames per second in output video
    dpi : int
        DPI for rendering
    output_path : str, optional
        Output file path. If None, auto-generates in current directory.
    
    Returns
    -------
    Path
        Path to the generated video file
    """
    print(f"\n{'='*60}")
    print(f"Generating NS video for {commo_ticker}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    futures_data = load_calibration_data(
        commo_ticker=commo_ticker,
        start_date=start_date,
        end_date=end_date,
    )
    
    if futures_data.empty:
        raise ValueError(f"No data loaded for {commo_ticker}")
    
    # Get unique dates
    dates = pd.DatetimeIndex(pd.to_datetime(futures_data.index)).unique().sort_values()
    
    # Apply frame stride
    dates = dates[::frame_stride]
    n_frames = len(dates)
    print(f"Total frames to render: {n_frames} (stride={frame_stride})")
    
    # Compute fixed y-axis range from observed prices
    print("Computing fixed axis ranges...")
    all_prices = futures_data["price"].astype(float).to_numpy()
    all_prices = all_prices[np.isfinite(all_prices) & (all_prices > 0)]
    
    y_lo = float(np.quantile(all_prices, 0.01))
    y_hi = float(np.quantile(all_prices, 0.999))
    y_pad = 0.1 * (y_hi - y_lo) if y_hi > y_lo else 1.0
    y_range = (0, y_hi + y_pad + 10)  # Y-axis minimum at 0, add extra +10 to upper bound
    
    print(f"Y-axis range: [{y_range[0]:.2f}, {y_range[1]:.2f}]")
    print(f"X-axis range: [{min_maturity}, {max_maturity}]")
    
    # NS evaluation grid starting from min_maturity
    ns_grid = np.arange(min_maturity, max_maturity + grid_step, grid_step)
    
    # Setup figure with timeline - clean dark style
    fig = plt.figure(figsize=(12, 8), dpi=dpi, facecolor='#000000')
    gs = fig.add_gridspec(2, 1, height_ratios=[10, 1], hspace=0.15)
    
    # Main plot - dark theme with fine grid
    ax = fig.add_subplot(gs[0], facecolor='#000000')
    ax.set_xlim(min_maturity, max_maturity)
    ax.set_ylim(*y_range)
    ax.set_xlabel("Time to Maturity (Business Days)", fontsize=11, color='#AAAAAA')
    ax.set_ylabel("Futures Price", fontsize=11, color='#AAAAAA')
    ax.grid(True, alpha=0.35, color='#4a4a4a', linestyle='-', linewidth=0.5)
    ax.tick_params(colors='#AAAAAA', which='both', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#4a4a4a')
        spine.set_linewidth(1)
    
    # Timeline axis - clean style
    ax_timeline = fig.add_subplot(gs[1], facecolor='#000000')
    ax_timeline.set_ylim(0, 1)
    ax_timeline.set_xlim(0, n_frames - 1)
    ax_timeline.set_yticks([])
    ax_timeline.spines['top'].set_visible(False)
    ax_timeline.spines['right'].set_visible(False)
    ax_timeline.spines['left'].set_visible(False)
    ax_timeline.spines['bottom'].set_edgecolor('#555555')
    ax_timeline.spines['bottom'].set_linewidth(1)
    ax_timeline.tick_params(colors='#999999', which='both', labelsize=8)
    
    # Create adaptive quarter/year markers to avoid clutter
    timeline_points = []
    for i, d in enumerate(dates):
        quarter = (d.month - 1) // 3 + 1  # 1-4
        timeline_points.append((i, d.year, quarter))

    # Keep first occurrence per (year, quarter)
    first_occurrence = {}
    for idx, year, quarter in timeline_points:
        key = (year, quarter)
        if key not in first_occurrence:
            first_occurrence[key] = idx

    entries = sorted((idx, year, quarter) for (year, quarter), idx in first_occurrence.items())

    max_ticks = 14
    selected = entries
    if len(selected) > max_ticks:
        # Keep only Q1 per year if too dense
        selected = [(idx, year, 1) for idx, year, quarter in entries if quarter == 1]
    if len(selected) > max_ticks and selected:
        step = math.ceil(len(selected) / max_ticks)
        selected = selected[::step]

    time_positions = [idx for idx, _, _ in selected]
    time_labels = [f"{year}" if quarter == 1 else f"Q{quarter}" for _, year, quarter in selected]

    ax_timeline.set_xticks(time_positions)
    ax_timeline.set_xticklabels(time_labels, fontsize=9, color='#AAAAAA')
    
    # Timeline cursor (triangle pointing down)
    timeline_cursor = ax_timeline.plot([0], [0.1], marker='v', color='#FFA500', 
                                       markersize=10, zorder=3)[0]
    
    # Initialize empty plots - orange/yellow style
    line_ns, = ax.plot([], [], color='#FFA500', linewidth=1.5, label='NS Fit')  # Orange, thinner
    scatter_actual = ax.scatter([], [], marker='x', s=50, c='#FFA500', 
                                linewidths=1.5, label='Actual', zorder=3)  # X markers
    
    # Text annotations - clean style
    date_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       fontsize=11, verticalalignment='top', color='#DDDDDD',
                       bbox=dict(boxstyle='round', facecolor='#000000', 
                                edgecolor='#555555', linewidth=1, alpha=0.8))
    
    params_text = ax.text(0.98, 0.98, '', transform=ax.transAxes,
                         fontsize=9, verticalalignment='top', horizontalalignment='right',
                         family='monospace', color='#DDDDDD',
                         bbox=dict(boxstyle='round', facecolor='#000000', 
                                  edgecolor='#555555', linewidth=1, alpha=0.8))
    
    legend = ax.legend(loc='lower right', fontsize=9, facecolor='#000000', 
                       edgecolor='#555555', labelcolor='#DDDDDD', framealpha=0.8)
    
    # Pre-compute NS params for all frames (with progress bar)
    print("\nPre-fitting NS curves...")
    params_cache = {}
    
    for date in tqdm(dates, desc="Fitting NS"):
        try:
            day_data = futures_data.loc[date]
            if isinstance(day_data, pd.Series):
                day_data = day_data.to_frame().T
            
            # Filter to plotting window
            day_data = day_data[
                (day_data["time_to_maturity"] >= min_maturity) &
                (day_data["time_to_maturity"] <= max_maturity)
            ].sort_values("time_to_maturity")
            
            if day_data.empty:
                params_cache[date] = None
                continue
            
            # Fit NS only on contracts >= exclude_short_for_fit
            fit_mask = day_data["time_to_maturity"] >= exclude_short_for_fit
            fit_data = day_data[fit_mask]
            
            if len(fit_data) == 0:
                params_cache[date] = None
                continue
            
            m_fit = fit_data["time_to_maturity"].to_numpy()
            p_fit = fit_data["price"].to_numpy()
            
            params = NS_estimation(m_fit, p_fit, verbosity=False)
            params_cache[date] = params
            
        except (KeyError, ValueError):
            params_cache[date] = None
    
    print(f"Fitted {sum(p is not None for p in params_cache.values())} / {len(dates)} dates")
    
    # Animation update function
    def update_frame(frame_idx: int):
        date = dates[frame_idx]
        
        # Update timeline cursor
        timeline_cursor.set_data([frame_idx], [0.1])
        
        try:
            day_data = futures_data.loc[date]
            if isinstance(day_data, pd.Series):
                day_data = day_data.to_frame().T
            
            day_data = day_data[
                (day_data["time_to_maturity"] >= min_maturity) &
                (day_data["time_to_maturity"] <= max_maturity)
            ].sort_values("time_to_maturity")
            
            if not day_data.empty:
                m_actual = day_data["time_to_maturity"].to_numpy()
                p_actual = day_data["price"].to_numpy()
                scatter_actual.set_offsets(np.c_[m_actual, p_actual])
            else:
                scatter_actual.set_offsets(np.empty((0, 2)))
            
            # Get cached params
            params = params_cache.get(date)
            
            if params is not None:
                # Evaluate NS on grid starting from min_maturity
                ns_log = NS_rate(ns_grid, params)
                ns_prices = np.exp(ns_log)
                line_ns.set_data(ns_grid, ns_prices)
                
                # Parameter text (4-parameter NS model)
                b0, b1, b2, lam = params
                params_str = (
                    f"β₀ = {b0:>7.4f}\n"
                    f"β₁ = {b1:>7.4f}\n"
                    f"β₂ = {b2:>7.4f}\n"
                    f"λ = {lam:>7.2f}"
                )
                params_text.set_text(params_str)
            else:
                line_ns.set_data([], [])
                params_text.set_text("No fit")
            
            # Update date
            date_text.set_text(f"{commo_ticker} — {date.date()}")
            
        except (KeyError, ValueError) as e:
            # Handle missing data gracefully
            line_ns.set_data([], [])
            scatter_actual.set_offsets(np.empty((0, 2)))
            date_text.set_text(f"{commo_ticker} — {date.date()} (no data)")
            params_text.set_text("")
        
        return line_ns, scatter_actual, date_text, params_text, timeline_cursor
    
    # Generate output path in videos folder
    videos_dir = ROOT_DIR / "videos"
    videos_dir.mkdir(exist_ok=True)
    
    if output_path is None:
        output_filename = f"ns_video_{commo_ticker}_{start_date}_{end_date}_stride{frame_stride}.mp4"
        output_path = videos_dir / output_filename
    else:
        output_path = Path(output_path)
    
    output_path = output_path.resolve()
    
    # Try FFMpeg first, fallback to Pillow
    if FFMpegWriter.isAvailable():
        writer_class = FFMpegWriter
        writer_name = "FFmpeg"
    elif PillowWriter.isAvailable():
        writer_class = PillowWriter
        writer_name = "Pillow (GIF)"
        # Change extension to .gif for PillowWriter
        if output_path.suffix.lower() == '.mp4':
            output_path = output_path.with_suffix('.gif')
            print("\nNote: FFmpeg not available, generating GIF instead of MP4")
    else:
        raise RuntimeError(
            "No video writer available. Install FFmpeg or Pillow.\n"
            "  FFmpeg: Close terminal, reopen, and try again\n"
            "  Pillow: pip install pillow"
        )
    
    writer_class = writer_class
    writer_name = writer_name
    
    print(f"\nUsing writer: {writer_name}")
    print(f"Rendering video to: {output_path}")
    print(f"Settings: {n_frames} frames @ {fps} fps, dpi={dpi}")
    
    anim = FuncAnimation(
        fig,
        update_frame,
        frames=n_frames,
        interval=1000/fps,
        blit=True,
        repeat=False,
    )
     
    # Save with progress bar
    writer = writer_class(fps=fps, metadata={'artist': 'NS Video Generator'})
    
    with tqdm(total=n_frames, desc="Encoding video") as pbar:
        def progress_callback(i, n):
            pbar.update(1)
        
        anim.save(str(output_path), writer=writer, dpi=dpi, progress_callback=progress_callback)
    
    plt.close(fig)
    
    print(f"\n✓ Video saved: {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Open in browser
    file_url = output_path.as_uri()
    print(f"\nOpening in browser: {file_url}")
    webbrowser.open(file_url)
    
    return output_path


if __name__ == "__main__":
    # Example usage

    for ticker in ['CC', 'KC', 'SB', 'CT']:
         video_path = make_ns_video(
            commo_ticker=ticker,
            start_date="1989-01-01",
            end_date="2025-12-31",
            min_maturity=10,
            max_maturity=500,
            exclude_short_for_fit=20,
            grid_step=15,
            frame_stride=1,  # Every date for full rendering
            fps=60,
            dpi=140,
        )
         
         print(f"\n{'='*60}")
         print(f"Done! Video path: {video_path}")
         print(f"{'='*60}")

    
    print(f"\n{'='*60}")
    print(f"Done! Video path: {video_path}")
    print(f"{'='*60}")
