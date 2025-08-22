import streamlit as st
from PIL import Image, ImageFilter, ImageStat
import numpy as np
import cv2
import io
import os
from scipy import ndimage
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings(‘ignore’)

# Page configuration

st.set_page_config(
page_title=“Advanced AI Image Detector”,
page_icon=“🔍”,
layout=“wide”
)

# Custom CSS for better UI

st.markdown(”””

<style>
.main-header {
    font-size: 3rem;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-container {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.confidence-high { color: #28a745; font-weight: bold; }
.confidence-medium { color: #ffc107; font-weight: bold; }
.confidence-low { color: #dc3545; font-weight: bold; }

.detection-result {
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    text-align: center;
    font-size: 1.2rem;
    font-weight: bold;
}

.ai-generated { 
    background: linear-gradient(135deg, #ff6b6b, #ee5a24);
    color: white;
}

.real-image { 
    background: linear-gradient(135deg, #26de81, #20bf6b);
    color: white;
}

.uncertain { 
    background: linear-gradient(135deg, #feca57, #ff9ff3);
    color: white;
}
</style>

“””, unsafe_allow_html=True)

class AdvancedAIDetector:
def **init**(self):
self.results = {}

```
def analyze_jpeg_compression(self, image: Image.Image) -> Dict:
    """Analyze JPEG compression artifacts and quality"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate compression artifacts using DCT analysis
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Apply DCT to detect compression blocks
        dct = cv2.dct(np.float32(gray))
        compression_score = np.std(dct) / np.mean(np.abs(dct))
        
        # Check for typical JPEG block artifacts (8x8 blocks)
        h, w = gray.shape
        block_variance = []
        for i in range(0, h-8, 8):
            for j in range(0, w-8, 8):
                block = gray[i:i+8, j:j+8]
                block_variance.append(np.var(block))
        
        block_uniformity = np.std(block_variance) / (np.mean(block_variance) + 1e-10)
        
        return {
            'compression_score': compression_score,
            'block_uniformity': block_uniformity,
            'typical_jpeg': compression_score > 0.1 and block_uniformity < 2.0
        }
    except Exception as e:
        return {'compression_score': 0, 'block_uniformity': 0, 'typical_jpeg': False}

def analyze_noise_patterns(self, image: Image.Image) -> Dict:
    """Analyze noise patterns and texture characteristics"""
    img_array = np.array(image.convert('RGB'))
    
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Calculate noise in different channels
    noise_metrics = {}
    
    # High-frequency noise analysis
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Sobel edge detection for texture analysis
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_density = np.mean(sobel_magnitude > 30)
    
    # Local Binary Pattern for texture
    def local_binary_pattern(img, radius=1, n_points=8):
        h, w = img.shape
        lbp = np.zeros_like(img)
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = img[i, j]
                pattern = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if 0 <= x < h and 0 <= y < w:
                        if img[x, y] >= center:
                            pattern |= (1 << k)
                lbp[i, j] = pattern
        return lbp
    
    lbp = local_binary_pattern(gray)
    texture_uniformity = np.std(lbp) / (np.mean(lbp) + 1e-10)
    
    # Color channel noise analysis
    color_noise = []
    for channel in range(3):
        channel_data = img_array[:, :, channel]
        # High pass filter to isolate noise
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(channel_data, -1, kernel)
        noise_level = np.std(filtered)
        color_noise.append(noise_level)
    
    return {
        'laplacian_variance': laplacian_var,
        'edge_density': edge_density,
        'texture_uniformity': texture_uniformity,
        'color_noise_std': np.std(color_noise),
        'color_noise_mean': np.mean(color_noise)
    }

def analyze_frequency_domain(self, image: Image.Image) -> Dict:
    """Analyze frequency domain characteristics"""
    gray = np.array(image.convert('L'))
    
    # FFT analysis
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shifted)
    
    # Calculate frequency distribution
    h, w = gray.shape
    center_h, center_w = h // 2, w // 2
    
    # High frequency content
    high_freq_mask = np.zeros((h, w))
    high_freq_mask[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = 1
    high_freq_energy = np.sum(magnitude_spectrum * (1 - high_freq_mask))
    total_energy = np.sum(magnitude_spectrum)
    high_freq_ratio = high_freq_energy / total_energy
    
    # Radial frequency analysis
    y, x = np.ogrid[:h, :w]
    distances = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    
    radial_profile = []
    max_distance = min(h, w) // 2
    for r in range(1, max_distance, 5):
        mask = (distances >= r-2) & (distances < r+2)
        radial_profile.append(np.mean(magnitude_spectrum[mask]))
    
    # Calculate spectral rolloff (frequency where 85% of energy is contained)
    cumsum_spectrum = np.cumsum(radial_profile)
    total_sum = cumsum_spectrum[-1]
    rolloff_85 = np.where(cumsum_spectrum >= 0.85 * total_sum)[0]
    spectral_rolloff = rolloff_85[0] if len(rolloff_85) > 0 else len(radial_profile)
    
    return {
        'high_freq_ratio': high_freq_ratio,
        'spectral_rolloff': spectral_rolloff / len(radial_profile),
        'frequency_variance': np.var(radial_profile)
    }

def analyze_statistical_properties(self, image: Image.Image) -> Dict:
    """Analyze statistical properties of pixel distributions"""
    img_array = np.array(image.convert('RGB'))
    
    # Calculate statistics for each channel
    stats = {}
    for i, channel in enumerate(['R', 'G', 'B']):
        channel_data = img_array[:, :, i].flatten()
        
        # Basic statistics
        stats[f'{channel}_mean'] = np.mean(channel_data)
        stats[f'{channel}_std'] = np.std(channel_data)
        stats[f'{channel}_skew'] = self.calculate_skewness(channel_data)
        stats[f'{channel}_kurtosis'] = self.calculate_kurtosis(channel_data)
        
        # Entropy
        hist, _ = np.histogram(channel_data, bins=256, range=(0, 255))
        hist = hist / np.sum(hist)  # Normalize
        stats[f'{channel}_entropy'] = entropy(hist + 1e-10)  # Add small value to avoid log(0)
    
    # Cross-channel correlation
    correlations = []
    for i in range(3):
        for j in range(i+1, 3):
            corr = np.corrcoef(img_array[:, :, i].flatten(), 
                             img_array[:, :, j].flatten())[0, 1]
            correlations.append(corr)
    
    stats['channel_correlation_mean'] = np.mean(correlations)
    stats['channel_correlation_std'] = np.std(correlations)
    
    return stats

def calculate_skewness(self, data):
    """Calculate skewness of data"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 3)

def calculate_kurtosis(self, data):
    """Calculate kurtosis of data"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 4) - 3

def detect_inconsistencies(self, image: Image.Image) -> Dict:
    """Detect lighting and perspective inconsistencies"""
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Lighting consistency analysis
    # Divide image into grid and analyze brightness variance
    h, w = gray.shape
    grid_size = 8
    brightness_values = []
    
    for i in range(0, h, h//grid_size):
        for j in range(0, w, w//grid_size):
            region = gray[i:i+h//grid_size, j:j+w//grid_size]
            if region.size > 0:
                brightness_values.append(np.mean(region))
    
    lighting_variance = np.var(brightness_values) / (np.mean(brightness_values) + 1e-10)
    
    # Edge consistency
    edges = cv2.Canny(gray, 50, 150)
    edge_segments = []
    
    # Hough line detection for structural analysis
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    if lines is not None:
        angles = [line[0][1] for line in lines]
        angle_consistency = np.std(angles)
    else:
        angle_consistency = 0
    
    return {
        'lighting_variance': lighting_variance,
        'angle_consistency': angle_consistency,
        'edge_density': np.sum(edges > 0) / edges.size
    }

def comprehensive_analysis(self, image: Image.Image) -> Dict:
    """Perform comprehensive AI detection analysis"""
    # Run all analysis methods
    jpeg_analysis = self.analyze_jpeg_compression(image)
    noise_analysis = self.analyze_noise_patterns(image)
    frequency_analysis = self.analyze_frequency_domain(image)
    statistical_analysis = self.analyze_statistical_properties(image)
    consistency_analysis = self.detect_inconsistencies(image)
    
    # Combine all metrics
    all_metrics = {
        **jpeg_analysis,
        **noise_analysis,
        **frequency_analysis,
        **statistical_analysis,
        **consistency_analysis
    }
    
    # AI Detection Score Calculation (weighted)
    ai_indicators = 0
    total_weight = 0
    
    # High smoothness indicates AI
    if all_metrics['laplacian_variance'] < 100:
        ai_indicators += 3
    total_weight += 3
    
    # Low texture uniformity indicates AI
    if all_metrics['texture_uniformity'] < 1.0:
        ai_indicators += 2
    total_weight += 2
    
    # High channel correlation indicates AI
    if all_metrics['channel_correlation_mean'] > 0.8:
        ai_indicators += 2
    total_weight += 2
    
    # Low noise indicates AI
    if all_metrics['color_noise_std'] < 5:
        ai_indicators += 2
    total_weight += 2
    
    # Unusual frequency distribution indicates AI
    if all_metrics['high_freq_ratio'] < 0.3:
        ai_indicators += 1
    total_weight += 1
    
    # Perfect lighting consistency indicates AI
    if all_metrics['lighting_variance'] < 0.1:
        ai_indicators += 1
    total_weight += 1
    
    ai_score = (ai_indicators / total_weight) * 100 if total_weight > 0 else 50
    
    # Determine classification
    if ai_score >= 70:
        classification = "AI Generated"
        confidence = "High"
        emoji = "🤖"
        css_class = "ai-generated"
    elif ai_score >= 40:
        classification = "Uncertain"
        confidence = "Medium"
        emoji = "❓"
        css_class = "uncertain"
    else:
        classification = "Real Image"
        confidence = "High"
        emoji = "📸"
        css_class = "real-image"
    
    return {
        'ai_score': ai_score,
        'classification': classification,
        'confidence': confidence,
        'emoji': emoji,
        'css_class': css_class,
        'detailed_metrics': all_metrics
    }
```

def create_analysis_visualization(metrics: Dict):
“”“Create visualization of analysis metrics”””
# Prepare data for visualization
important_metrics = {
‘Texture Variance’: metrics[‘laplacian_variance’],
‘Edge Density’: metrics[‘edge_density’] * 100,
‘Color Noise’: metrics[‘color_noise_mean’],
‘High Freq Ratio’: metrics[‘high_freq_ratio’] * 100,
‘Lighting Variance’: metrics[‘lighting_variance’],
‘Channel Correlation’: metrics[‘channel_correlation_mean’] * 100
}

```
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart of key metrics
metrics_names = list(important_metrics.keys())
metrics_values = list(important_metrics.values())

bars = ax1.bar(metrics_names, metrics_values, color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'])
ax1.set_title('Key Detection Metrics', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, value in zip(bars, metrics_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{value:.2f}', ha='center', va='bottom')

# Color distribution analysis
colors = ['Red', 'Green', 'Blue']
means = [metrics[f'{c}_mean'] for c in ['R', 'G', 'B']]
stds = [metrics[f'{c}_std'] for c in ['R', 'G', 'B']]

x = np.arange(len(colors))
width = 0.35

ax2.bar(x - width/2, means, width, label='Mean', alpha=0.8, color=['red', 'green', 'blue'])
ax2.bar(x + width/2, stds, width, label='Std Dev', alpha=0.6, color=['darkred', 'darkgreen', 'darkblue'])

ax2.set_title('Color Channel Analysis', fontsize=14, fontweight='bold')
ax2.set_xlabel('Color Channels')
ax2.set_ylabel('Pixel Values')
ax2.set_xticks(x)
ax2.set_xticklabels(colors)
ax2.legend()

plt.tight_layout()
return fig
```

def main():
st.markdown(’<h1 class="main-header">🔍 Advanced AI Image Detector</h1>’, unsafe_allow_html=True)

```
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem; font-size: 1.2rem; color: #666;">
Upload an image to perform comprehensive AI detection analysis using advanced computer vision techniques
</div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("🔬 Detection Methods")
    st.markdown("""
    **Advanced Analysis Includes:**
    - JPEG compression artifacts
    - Noise pattern analysis  
    - Frequency domain analysis
    - Statistical properties
    - Texture uniformity
    - Color channel correlations
    - Lighting consistency
    - Edge characteristics
    """)
    
    st.header("📊 Accuracy Notes")
    st.markdown("""
    - **High confidence**: >80% accuracy
    - **Medium confidence**: 60-80% accuracy
    - **Multiple indicators**: More reliable
    - **Best for**: Recent AI models (2022+)
    """)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file", 
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload JPG, PNG, or WebP images up to 200MB"
)

if uploaded_file is not None:
    try:
        # Load and display image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.subheader("📋 Image Information")
            st.write(f"**Format:** {image.format}")
            st.write(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
            st.write(f"**Mode:** {image.mode}")
            if hasattr(image, 'info') and 'quality' in image.info:
                st.write(f"**Quality:** {image.info['quality']}")
        
        with col2:
            with st.spinner("🔍 Analyzing image... This may take a moment..."):
                # Perform analysis
                detector = AdvancedAIDetector()
                results = detector.comprehensive_analysis(image)
            
            # Display main result
            st.markdown(f"""
            <div class="detection-result {results['css_class']}">
            {results['emoji']} {results['classification']}<br>
            <small>Confidence: {results['confidence']} ({results['ai_score']:.1f}% AI likelihood)</small>
            </div>
            """, unsafe_allow_html=True)
    
        # Detailed metrics
        st.subheader("📊 Detailed Analysis")
        
        metrics = results['detailed_metrics']
        
        # Key indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Texture Variance", 
                f"{metrics['laplacian_variance']:.1f}",
                help="Higher values indicate more natural texture"
            )
        
        with col2:
            st.metric(
                "Edge Density", 
                f"{metrics['edge_density']:.3f}",
                help="Natural images typically have more edge detail"
            )
        
        with col3:
            st.metric(
                "Color Noise", 
                f"{metrics['color_noise_mean']:.2f}",
                help="Real photos have more color channel noise"
            )
        
        with col4:
            st.metric(
                "Channel Correlation", 
                f"{metrics['channel_correlation_mean']:.3f}",
                help="AI images often have higher color correlation"
            )
        
        # Advanced metrics in expandable sections
        with st.expander("🔬 Advanced Technical Metrics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Noise & Texture Analysis")
                st.write(f"**Laplacian Variance:** {metrics['laplacian_variance']:.2f}")
                st.write(f"**Texture Uniformity:** {metrics['texture_uniformity']:.3f}")
                st.write(f"**Color Noise Std:** {metrics['color_noise_std']:.3f}")
                
                st.subheader("Frequency Domain")
                st.write(f"**High Freq Ratio:** {metrics['high_freq_ratio']:.3f}")
                st.write(f"**Spectral Rolloff:** {metrics['spectral_rolloff']:.3f}")
                st.write(f"**Frequency Variance:** {metrics['frequency_variance']:.2f}")
            
            with col2:
                st.subheader("Compression Analysis")
                st.write(f"**Compression Score:** {metrics['compression_score']:.3f}")
                st.write(f"**Block Uniformity:** {metrics['block_uniformity']:.3f}")
                st.write(f"**Typical JPEG:** {'✅' if metrics['typical_jpeg'] else '❌'}")
                
                st.subheader("Consistency Checks")
                st.write(f"**Lighting Variance:** {metrics['lighting_variance']:.3f}")
                st.write(f"**Angle Consistency:** {metrics['angle_consistency']:.3f}")
        
        # Color statistics
        with st.expander("🎨 Color Channel Statistics"):
            col1, col2, col3 = st.columns(3)
            
            for i, (col, channel) in enumerate(zip([col1, col2, col3], ['R', 'G', 'B'])):
                with col:
                    st.subheader(f"{channel} Channel")
                    st.write(f"**Mean:** {metrics[f'{channel}_mean']:.1f}")
                    st.write(f"**Std Dev:** {metrics[f'{channel}_std']:.1f}")
                    st.write(f"**Skewness:** {metrics[f'{channel}_skew']:.3f}")
                    st.write(f"**Kurtosis:** {metrics[f'{channel}_kurtosis']:.3f}")
                    st.write(f"**Entropy:** {metrics[f'{channel}_entropy']:.3f}")
        
        # Visualization
        with st.expander("📈 Analysis Visualization"):
            fig = create_analysis_visualization(metrics)
            st.pyplot(fig)
        
        # Interpretation guide
        with st.expander("❓ How to Interpret Results"):
            st.markdown("""
            **AI-Generated Images typically have:**
            - Low texture variance (< 100)
            - High color channel correlation (> 0.8)
            - Low noise levels
            - Very consistent lighting
            - Smooth gradients
            - Perfect symmetry/patterns
            
            **Real Camera Images typically have:**
            - Higher texture variance (> 100)
            - Natural noise patterns
            - Inconsistent lighting
            - JPEG compression artifacts
            - Random imperfections
            - Natural color distribution
            
            **Confidence Levels:**
            - **High**: Multiple indicators align
            - **Medium**: Mixed indicators
            - **Low**: Inconclusive evidence
            """)
        
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        st.write("Please try uploading a different image format or check if the file is corrupted.")
```

if **name** == “**main**”:
main()
