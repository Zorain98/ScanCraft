import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
import io
from datetime import datetime
import zipfile

# Configure page
st.set_page_config(
    page_title="ScanScript - Shipment ID Extractor",
    page_icon="📦",
    layout="wide"
)

# Force light theme
st.markdown("""
<style>
    .stApp {
        color-scheme: light;
    }
    
    /* Force light theme colors */
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp {
        background-color: white;
        color: black;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #ff6b6b;
        color: white;
        border: none;
    }
    
    .stButton > button:hover {
        background-color: #28a745;
        color: white;
    }
    
    /* Metrics and info boxes */
    .metric-container {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
    }
    
    /* Progress bar */
    .stProgress .st-bo {
        background-color: #28a745;
    }
    
    /* Success/Error messages */
    .stAlert {
        background-color: #d4edda;
        color: #155724;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: white !important;
        color: black !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #333333 !important;
    }
    
    /* Text */
    p, div, span {
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)
def extract_text_from_image(image):
    """Extract text from image using OCR"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if it's colored
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply some preprocessing to improve OCR accuracy
        # Increase contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (1, 1), 0)
        
        # Apply threshold to get black and white image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use pytesseract to extract text
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        
        return text
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return ""

def extract_shipment_id(text):
    """Extract shipment ID from text using regex patterns"""
    # Multiple patterns to catch different formats
    patterns = [
        r'Shipment ID\s*[\n\r]*\s*(\d+[,\d]*)',  # "Shipment ID" followed by numbers
        r'shipment id\s*[\n\r]*\s*(\d+[,\d]*)',  # case insensitive
        r'ID\s*[\n\r]*\s*(\d+[,\d]*)',           # just "ID" followed by numbers
        r'(\d{3}[,\d]{3,})',                      # Pattern for numbers with commas (like 101,831)
        r'(\d{6,})',                              # 6 or more consecutive digits
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Clean the match (remove commas)
            shipment_id = matches[0].replace(',', '')
            # Validate that it looks like a reasonable shipment ID
            if len(shipment_id) >= 3 and shipment_id.isdigit():
                return shipment_id
    
    return None

def main():
    st.title("📦 Shipment ID Extractor")
    st.markdown("Upload images containing shipment details and extract Shipment IDs to CSV")
    
    # Sidebar for instructions
    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. Upload one or more images containing shipment details
        2. The app will automatically extract Shipment IDs using OCR
        3. Review the extracted data
        4. Download the results as CSV
        
        **Supported formats:** PNG, JPG, JPEG
        """)
        
        st.header("Tips for better results")
        st.markdown("""
        - Use clear, high-resolution images
        - Ensure good lighting and contrast
        - Avoid blurry or rotated images
        """)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload images containing shipment details"
    )
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} image(s)")
        
        # Process button
        if st.button("Extract Shipment IDs", type="primary"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create columns for displaying results
            col1, col2 = st.columns([1, 2])
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                try:
                    # Open and display image
                    image = Image.open(uploaded_file)
                    
                    with col1:
                        if i == 0:  # Show header only once
                            st.subheader("Sample Images")
                        st.image(image, caption=uploaded_file.name, width=200)
                    
                    # Extract text from image
                    extracted_text = extract_text_from_image(image)
                    
                    # Extract shipment ID
                    shipment_id = extract_shipment_id(extracted_text)
                    
                    # Store results
                    result = {
                        'File Name': uploaded_file.name,
                        'Shipment ID': shipment_id if shipment_id else 'Not Found',
                        'Status': 'Success' if shipment_id else 'Failed',
                        'Extracted Text Preview': extracted_text[:100] + '...' if len(extracted_text) > 100 else extracted_text
                    }
                    results.append(result)
                    
                except Exception as e:
                    results.append({
                        'File Name': uploaded_file.name,
                        'Shipment ID': 'Error',
                        'Status': 'Error',
                        'Extracted Text Preview': f'Error: {str(e)}'
                    })
            
            status_text.text("Processing complete!")
            progress_bar.progress(1.0)
            
            # Display results
            with col2:
                st.subheader("Extraction Results")
                df = pd.DataFrame(results)
                
                # Color code the status
                def color_status(val):
                    if val == 'Success':
                        return 'background-color: #d4edda'
                    elif val == 'Failed':
                        return 'background-color: #f8d7da'
                    else:
                        return 'background-color: #fff3cd'
                
                styled_df = df.style.applymap(color_status, subset=['Status'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Statistics
                success_count = len([r for r in results if r['Status'] == 'Success'])
                st.metric("Successfully Extracted", f"{success_count}/{len(results)}")
                
                # Download options
                st.subheader("Download Results")
                
                # Prepare CSV data
                csv_data = []
                for result in results:
                    if result['Shipment ID'] != 'Not Found' and result['Shipment ID'] != 'Error':
                        csv_data.append({
                            'Shipment ID': result['Shipment ID'],
                            'Source File': result['File Name'],
                            'Extracted At': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                
                if csv_data:
                    csv_df = pd.DataFrame(csv_data)
                    csv_buffer = io.StringIO()
                    csv_df.to_csv(csv_buffer, index=False)
                    csv_string = csv_buffer.getvalue()
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.download_button(
                            label="📥 Download CSV (Shipment IDs only)",
                            data=csv_string,
                            file_name=f"shipment_ids_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col_b:
                        # Full results CSV
                        full_csv_buffer = io.StringIO()
                        df.to_csv(full_csv_buffer, index=False)
                        full_csv_string = full_csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download Full Results",
                            data=full_csv_string,
                            file_name=f"full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("No valid Shipment IDs found to download.")
                
                # Show extracted shipment IDs
                if csv_data:
                    st.subheader("Extracted Shipment IDs")
                    shipment_ids = [item['Shipment ID'] for item in csv_data]
                    st.write(", ".join(shipment_ids))

    else:
        st.info("Please upload image files to get started.")
        
        # Show example
        st.subheader("Example")
        st.markdown("""
        Upload images like the one you showed containing:
        - Portal Data Correction Request Details
        - Shipment ID numbers
        - Other shipment information
        
        The app will automatically detect and extract the Shipment ID values.
        """)

if __name__ == "__main__":
    # Check if pytesseract is available
    try:
        pytesseract.get_tesseract_version()
        main()
    except Exception as e:
        st.error("""
        **Tesseract OCR not found!**
        
        Please install Tesseract OCR:
        
        **Windows:**
        1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
        2. Install and add to PATH
        
        **macOS:**
        ```bash
        brew install tesseract
        ```
        
        **Linux:**
        ```bash
        sudo apt-get install tesseract-ocr
        ```
        
        **Python packages:**
        ```bash
        pip install pytesseract opencv-python pillow
        ```

        """)
