import streamlit as st
from PIL import Image
import numpy as np
import cv2
from scipy.stats import entropy
import matplotlib.pyplot as plt

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
def analyze_jpeg_compression(self, image):
    """Analyze JPEG compression artifacts"""
    try:
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Apply DCT to detect compression blocks
        dct = cv2.dct(np.float32(gray))
        compression_score = np.std(dct) / (np.mean(np.abs(dct)) + 1e-10)
        
        # Check for JPEG block artifacts
        h, w = gray.shape
        block_variance = []
        for i in range(0, h-8, 8):
            for j in range(0, w-8, 8):
                if i+8 < h and j+8 < w:
                    block = gray[i:i+8, j:j+8]
                    block_variance.append(np.var(block))
        
        if len(block_variance) > 0:
            block_uniformity = np.std(block_variance) / (np.mean(block_variance) + 1e-10)
        else:
            block_uniformity = 0
        
        return {
            'compression_score': compression_score,
            'block_uniformity': block_uniformity,
            'typical_jpeg': compression_score > 0.1 and block_uniformity < 2.0
        }
    except Exception:
        return {'compression_score': 0, 'block_uniformity': 0, 'typical_jpeg': False}

def analyze_noise_patterns(self, image):
    """Analyze noise patterns and texture"""
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # High-frequency noise analysis
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_density = np.mean(sobel_magnitude > 30)
    
    # Simple texture analysis
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filtered = cv2.filter2D(gray, -1, kernel)
    texture_variance = np.var(filtered)
    
    # Color noise analysis
    color_noise = []
    for channel in range(3):
        channel_data = img_array[:, :, channel]
        filtered_channel = cv2.filter2D(channel_data, -1, kernel)
        noise_level = np.std(filtered_channel)
        color_noise.append(noise_level)
    
    return {
        'laplacian_variance': laplacian_var,
        'edge_density': edge_density,
        'texture_variance': texture_variance,
        'color_noise_std': np.std(color_noise),
        'color_noise_mean': np.mean(color_noise)
    }

def analyze_frequency_domain(self, image):
    """Analyze frequency characteristics"""
    gray = np.array(image.convert('L'))
    
    # FFT analysis
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shifted)
    
    h, w = gray.shape
    center_h, center_w = h // 2, w // 2
    
    # High frequency content
    high_freq_mask = np.zeros((h, w))
    if h > 8 and w > 8:
        high_freq_mask[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = 1
        high_freq_energy = np.sum(magnitude_spectrum * (1 - high_freq_mask))
        total_energy = np.sum(magnitude_spectrum)
        high_freq_ratio = high_freq_energy / (total_energy + 1e-10)
    else:
        high_freq_ratio = 0.5
    
    return {
        'high_freq_ratio': high_freq_ratio,
        'frequency_variance': np.var(magnitude_spectrum)
    }

def analyze_color_statistics(self, image):
    """Analyze color channel statistics"""
    img_array = np.array(image.convert('RGB'))
    
    stats = {}
    correlations = []
    
    # Calculate stats for each channel
    for i, channel in enumerate(['R', 'G', 'B']):
        channel_data = img_array[:, :, i].flatten()
        stats[f'{channel}_mean'] = np.mean(channel_data)
        stats[f'{channel}_std'] = np.std(channel_data)
        
        # Entropy
        hist, _ = np.histogram(channel_data, bins=64, range=(0, 255))
        hist = hist / (np.sum(hist) + 1e-10)
        stats[f'{channel}_entropy'] = entropy(hist + 1e-10)
    
    # Channel correlations
    for i in range(3):
        for j in range(i+1, 3):
            corr = np.corrcoef(img_array[:, :, i].flatten(), 
                             img_array[:, :, j].flatten())[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    if len(correlations) > 0:
        stats['channel_correlation_mean'] = np.mean(correlations)
    else:
        stats['channel_correlation_mean'] = 0.5
        
    return stats

def detect_lighting_consistency(self, image):
    """Detect lighting inconsistencies"""
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    h, w = gray.shape
    grid_size = 8
    brightness_values = []
    
    # Analyze brightness in grid regions
    for i in range(0, h, h//grid_size):
        for j in range(0, w, w//grid_size):
            region_end_i = min(i + h//grid_size, h)
            region_end_j = min(j + w//grid_size, w)
            region = gray[i:region_end_i, j:region_end_j]
            if region.size > 0:
                brightness_values.append(np.mean(region))
    
    if len(brightness_values) > 1:
        lighting_variance = np.var(brightness_values) / (np.mean(brightness_values) + 1e-10)
    else:
        lighting_variance = 0.5
    
    return {'lighting_variance': lighting_variance}

def comprehensive_analysis(self, image):
    """Perform comprehensive AI detection"""
    try:
        # Run all analyses
        jpeg_analysis = self.analyze_jpeg_compression(image)
        noise_analysis = self.analyze_noise_patterns(image)
        frequency_analysis = self.analyze_frequency_domain(image)
        color_analysis = self.analyze_color_statistics(image)
        lighting_analysis = self.detect_lighting_consistency(image)
        
        # Combine metrics
        all_metrics = {
            **jpeg_analysis,
            **noise_analysis,
            **frequency_analysis,
            **color_analysis,
            **lighting_analysis
        }
        
        # Calculate AI score with weighted indicators
        ai_indicators = 0
        total_weight = 0
        
        # Texture smoothness (high weight)
        if all_metrics['laplacian_variance'] < 100:
            ai_indicators += 3
        total_weight += 3
        
        # Low texture variance indicates AI
        if all_metrics['texture_variance'] < 50:
            ai_indicators += 2
        total_weight += 2
        
        # High channel correlation indicates AI
        if all_metrics['channel_correlation_mean'] > 0.8:
            ai_indicators += 2
        total_weight += 2
        
        # Low noise indicates AI
        if all_metrics['color_noise_std'] < 10:
            ai_indicators += 2
        total_weight += 2
        
        # Frequency analysis
        if all_metrics['high_freq_ratio'] < 0.3:
            ai_indicators += 1
        total_weight += 1
        
        # Perfect lighting
        if all_metrics['lighting_variance'] < 0.1:
            ai_indicators += 1
        total_weight += 1
        
        # Calculate final score
        ai_score = (ai_indicators / total_weight * 100) if total_weight > 0 else 50
        
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
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None
```

def create_metrics_chart(metrics):
“”“Create simple metrics visualization”””
try:
important_metrics = {
‘Texture’: min(metrics[‘laplacian_variance’], 200),
‘Edge Density’: metrics[‘edge_density’] * 100,
‘Color Noise’: min(metrics[‘color_noise_mean’], 50),
‘High Freq’: metrics[‘high_freq_ratio’] * 100,
‘Lighting’: min(metrics[‘lighting_variance’] * 100, 100),
‘Correlation’: metrics[‘channel_correlation_mean’] * 100
}

```
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(important_metrics.keys())
    values = list(important_metrics.values())
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
    
    bars = ax.bar(names, values, color=colors)
    ax.set_title('Detection Metrics Analysis', fontsize=16, fontweight='bold')
    ax.set_ylabel('Metric Values')
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
except Exception:
    return None
```

def main():
st.markdown(’<h1 class="main-header">🔍 Advanced AI Image Detector</h1>’, unsafe_allow_html=True)

```
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem; font-size: 1.2rem; color: #666;">
Upload an image for comprehensive AI detection analysis
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔬 Analysis Methods")
    st.markdown("""
    **Detection Techniques:**
    - JPEG compression analysis
    - Noise pattern detection
    - Frequency domain analysis
    - Color statistics
    - Texture analysis
    - Lighting consistency
    """)
    
    st.header("📊 How to Read Results")
    st.markdown("""
    - **AI Generated (70-100%)**: Very likely artificial
    - **Uncertain (40-69%)**: Mixed indicators  
    - **Real Image (0-39%)**: Likely authentic
    """)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file", 
    type=["jpg", "jpeg", "png"],
    help="Upload JPG or PNG images"
)

if uploaded_file is not None:
    try:
        # Load image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.subheader("📋 Image Info")
            st.write(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
            st.write(f"**Format:** {image.format}")
            st.write(f"**Mode:** {image.mode}")
        
        with col2:
            with st.spinner("🔍 Analyzing image..."):
                # Perform analysis
                detector = AdvancedAIDetector()
                results = detector.comprehensive_analysis(image)
            
            if results:
                # Display main result
                st.markdown(f"""
                <div class="detection-result {results['css_class']}">
                {results['emoji']} {results['classification']}<br>
                <small>Confidence: {results['confidence']} ({results['ai_score']:.1f}% AI likelihood)</small>
                </div>
                """, unsafe_allow_html=True)
    
        if results:
            # Key metrics
            st.subheader("📊 Key Metrics")
            metrics = results['detailed_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Texture Variance", f"{metrics['laplacian_variance']:.1f}")
            
            with col2:
                st.metric("Edge Density", f"{metrics['edge_density']:.3f}")
            
            with col3:
                st.metric("Color Noise", f"{metrics['color_noise_mean']:.2f}")
            
            with col4:
                st.metric("Channel Correlation", f"{metrics['channel_correlation_mean']:.3f}")
            
            # Detailed analysis
            with st.expander("🔬 Detailed Technical Analysis"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Texture & Noise")
                    st.write(f"**Laplacian Variance:** {metrics['laplacian_variance']:.2f}")
                    st.write(f"**Texture Variance:** {metrics['texture_variance']:.2f}")
                    st.write(f"**Color Noise Std:** {metrics['color_noise_std']:.3f}")
                    st.write(f"**Edge Density:** {metrics['edge_density']:.3f}")
                
                with col2:
                    st.subheader("Frequency & Compression")
                    st.write(f"**High Freq Ratio:** {metrics['high_freq_ratio']:.3f}")
                    st.write(f"**Compression Score:** {metrics['compression_score']:.3f}")
                    st.write(f"**Block Uniformity:** {metrics['block_uniformity']:.3f}")
                    st.write(f"**Lighting Variance:** {metrics['lighting_variance']:.3f}")
            
            # Color analysis
            with st.expander("🎨 Color Channel Analysis"):
                col1, col2, col3 = st.columns(3)
                
                for i, (col, channel) in enumerate(zip([col1, col2, col3], ['R', 'G', 'B'])):
                    with col:
                        st.subheader(f"{channel} Channel")
                        st.write(f"**Mean:** {metrics[f'{channel}_mean']:.1f}")
                        st.write(f"**Std Dev:** {metrics[f'{channel}_std']:.1f}")
                        st.write(f"**Entropy:** {metrics[f'{channel}_entropy']:.3f}")
            
            # Visualization
            with st.expander("📈 Metrics Visualization"):
                fig = create_metrics_chart(metrics)
                if fig:
                    st.pyplot(fig)
            
            # Interpretation
            with st.expander("❓ How to Interpret Results"):
                st.markdown("""
                **🤖 AI-Generated Images usually have:**
                - Low texture variance (< 100)
                - High color correlation (> 0.8)
                - Very smooth appearance
                - Consistent lighting
                - Low noise levels
                
                **📸 Real Images usually have:**
                - Higher texture variance (> 100)
                - Natural noise and imperfections
                - JPEG compression artifacts
                - Variable lighting
                - Random details
                
                **❓ Uncertain Results:**
                - Mixed indicators
                - Heavily edited photos
                - Artistic images
                - Low resolution images
                """)
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.write("Please try a different image or check the file format.")
```

if **name** == “**main**”:
main()
