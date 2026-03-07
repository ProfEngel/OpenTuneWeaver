import os
import shutil
import json
import logging
import time
import gc
import sys
from pathlib import Path
from fpdf import FPDF
from PIL import Image, ImageEnhance
import requests

# ========================================
# LOAD CENTRAL CONFIGURATION
# ========================================
sys.path.append(str(Path(__file__).parent.parent.parent))  # To main directory
from config_loader import PipelineConfigLoader

# Load configuration for this module
config_loader = PipelineConfigLoader()
module_config = config_loader.get_module_config("01_convert")
pipeline_config = config_loader.get_pipeline_config()

# Extract configuration values
API_BASE_URL = module_config.get("api_base_url", "")
API_KEY = module_config.get("api_key", "")
MODEL_NAME = module_config.get("model_name", "")

CONTRAST_FACTOR = pipeline_config.get("contrast_factor", 2.0)
IMAGE_DESCRIPTION_TIMEOUT = pipeline_config.get("image_description_timeout", 60)
MAX_RETRIES = pipeline_config.get("max_retries", 3)
CREATE_COMBINED_MD = pipeline_config.get("create_combined_md", False)

# Show loaded configuration
print("=" * 60)
print("📋 CONFIGURATION LOADED (01_convert)")
print("=" * 60)
config_loader.print_config_summary()
print(f"  🖼️ Contrast Factor: {CONTRAST_FACTOR}")
print("=" * 60)

# Docling imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

# ========================================
# HELPER FUNCTIONS
# ========================================

def setup_logging():
    """Configures logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def print_section(title):
    """Prints a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def get_pdf_page_count(pdf_path):
    """Determines the page count of a PDF."""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            return len(pdf.pages)
    except:
        return 0

# ========================================
# 0. DIRECTORY SETUP
# ========================================

def setup_directories():
    """Creates all necessary directories."""
    directories = ['UPLOAD', 'INPUT', 'OUTPUT']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directory created/verified: {directory}")

# ========================================
# 1. DOCUMENT CONVERSION TO PDF
# ========================================

def convert_with_libreoffice(file_path, output_dir):
    """Converts files with LibreOffice to PDF."""
    try:
        result = os.system(f'soffice --headless --convert-to pdf --outdir {output_dir} "{file_path}"')
        if result == 0:
            print(f"✅ LibreOffice conversion successful: {file_path}")
        else:
            print(f"❌ LibreOffice conversion failed: {file_path}")
    except Exception as e:
        print(f"❌ Error in LibreOffice conversion of {file_path}: {e}")

def convert_txt_to_pdf(file_path, output_path):
    """Converts TXT files to PDF."""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    pdf.cell(200, 10, txt=line.strip(), ln=True)
                except:
                    pdf.cell(200, 10, txt=line.strip().encode('latin-1', 'ignore').decode('latin-1'), ln=True)
        
        pdf.output(str(output_path))
        print(f"✅ TXT to PDF converted: {output_path}")
    except Exception as e:
        print(f"❌ Error in TXT conversion of {file_path}: {e}")

def convert_html_to_pdf(file_path, output_path):
    """Converts HTML files to PDF with wkhtmltopdf."""
    try:
        result = os.system(f'wkhtmltopdf "{file_path}" "{output_path}"')
        if result == 0:
            print(f"✅ HTML to PDF converted: {output_path}")
        else:
            print(f"❌ HTML conversion failed: {file_path}")
    except Exception as e:
        print(f"❌ Error in HTML conversion of {file_path}: {e}")

def convert_documents_to_pdf():
    """Converts all documents in the UPLOAD directory to PDF."""
    print_section("1. DOCUMENT CONVERSION TO PDF")
    
    upload_dir = Path('UPLOAD')
    input_dir = Path('INPUT')
    
    if not upload_dir.exists() or not any(upload_dir.iterdir()):
        print("⚠️ No files found in UPLOAD directory.")
        return
    
    converted_count = 0
    
    for file_path in upload_dir.iterdir():
        if file_path.is_file():
            ext = file_path.suffix.lower()
            output_file = input_dir / (file_path.stem + '.pdf')
            
            print(f"🔄 Processing: {file_path.name}")
            
            if ext == '.pdf':
                shutil.copy(file_path, output_file)
                print(f"✅ PDF copied: {output_file}")
                converted_count += 1
            elif ext in ['.docx', '.pptx', '.xlsx', '.xml']:
                convert_with_libreoffice(file_path, input_dir)
                converted_count += 1
            elif ext == '.txt':
                convert_txt_to_pdf(file_path, output_file)
                converted_count += 1
            elif ext == '.html':
                convert_html_to_pdf(file_path, output_file)
                converted_count += 1
            else:
                print(f"⚠️ Format {ext} not supported: {file_path.name}")
    
    print(f"📊 Converted files: {converted_count}")

# ========================================
# 2. PDF TO MARKDOWN (WITHOUT VLM!)
# ========================================

def setup_standard_converter():
    """Configures the DocumentConverter for fast text extraction WITHOUT VLM."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0
    
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

def setup_basic_converter():
    """Creates a minimal converter as fallback."""
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption()
        }
    )

def extract_images_from_result(result, pdf_file, output_dir):
    """Extracts and saves images from the conversion result."""
    try:
        artifacts_dir = output_dir / f"{pdf_file.stem}_artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        extracted_images = 0
        
        if artifacts_dir.exists():
            existing_images = list(artifacts_dir.glob("*.png")) + list(artifacts_dir.glob("*.jpg"))
            if existing_images:
                print(f"📁 {len(existing_images)} images already in artifacts directory")
                return len(existing_images)
        
        if hasattr(result.document, 'pictures'):
            for idx, picture in enumerate(result.document.pictures):
                try:
                    image_filename = artifacts_dir / f"image_{idx:03d}.png"
                    with open(image_filename, 'wb') as f:
                        f.write(picture.data)
                    extracted_images += 1
                    print(f"   🖼️ Image extracted: {image_filename.name}")
                except Exception as e:
                    print(f"   ⚠️ Error extracting image {idx}: {e}")
        
        return extracted_images
        
    except Exception as e:
        print(f"❌ Error in image extraction: {e}")
        return 0

def process_remaining_pdfs(pdf_files_to_process):
    """Processes PDF files with optimized standard conversion."""
    if not pdf_files_to_process:
        print("✅ All PDFs already processed")
        return True
    
    print(f"🔄 Processing {len(pdf_files_to_process)} pending PDF files...")
    
    output_dir = Path("OUTPUT")
    successful_conversions = 0
    total_start_time = time.time()
    
    for pdf_file in pdf_files_to_process:
        start_time = time.time()
        page_count = get_pdf_page_count(pdf_file)
        
        print(f"\n🔄 Converting: {pdf_file.name}")
        if page_count > 0:
            print(f"📄 Page count: {page_count}")
        
        success = False
        
        try:
            print("🚀 Using optimized standard conversion...")
            doc_converter = setup_standard_converter()
            result = doc_converter.convert(pdf_file)
            
            md_filename = output_dir / f"{pdf_file.stem}.md"
            result.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)
            
            extracted_count = extract_images_from_result(result, pdf_file, output_dir)
            
            elapsed = time.time() - start_time
            print(f"✅ Conversion successful: {md_filename}")
            print(f"🖼️ Images extracted: {extracted_count}")
            print(f"⏱️ Time: {elapsed:.1f}s")
            if page_count > 0:
                print(f"📊 Performance: {elapsed/page_count:.2f}s per page")
            
            successful_conversions += 1
            success = True
            
        except Exception as e:
            print(f"❌ Standard conversion failed: {e}")
            
            try:
                print("🔄 Fallback: Basic conversion...")
                basic_converter = setup_basic_converter()
                result = basic_converter.convert(pdf_file)
                
                md_filename = output_dir / f"{pdf_file.stem}.md"
                
                try:
                    result.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)
                except:
                    markdown_content = result.document.export_to_markdown()
                    with open(md_filename, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                
                elapsed = time.time() - start_time
                print(f"✅ Basic conversion successful: {md_filename}")
                print(f"⏱️ Time: {elapsed:.1f}s")
                
                successful_conversions += 1
                success = True
                
            except Exception as basic_error:
                print(f"❌ Basic conversion also failed: {basic_error}")
                
                error_md = output_dir / f"{pdf_file.stem}_error.md"
                with open(error_md, 'w', encoding='utf-8') as f:
                    f.write(f"# {pdf_file.stem}\n\n**Conversion Failed**\n\n")
                    f.write(f"The PDF file could not be processed.\n")
                    f.write(f"Error: {str(basic_error)}")
                print(f"⚠️ Error markdown created: {error_md}")
    
    total_elapsed = time.time() - total_start_time
    print(f"\n📊 CONVERSION COMPLETED:")
    print(f"   - Successful: {successful_conversions}/{len(pdf_files_to_process)}")
    print(f"   - Total time: {total_elapsed:.1f}s")
    print(f"   - Average: {total_elapsed/len(pdf_files_to_process):.1f}s per file")
    
    gc.collect()
    print("✅ Memory freed")
    
    return successful_conversions > 0

# ========================================
# 3. STATUS CHECK
# ========================================

def check_processing_status():
    """Checks which processing steps are already completed."""
    print_section("🔍 STATUS CHECK")
    
    upload_dir = Path('UPLOAD')
    input_dir = Path('INPUT')
    output_dir = Path('OUTPUT')
    
    status = {
        'pdf_conversion_needed': False,
        'markdown_conversion_needed': False,
        'pdf_files_to_process': [],
        'existing_md_files': [],
        'existing_artifacts_dirs': []
    }
    
    if upload_dir.exists():
        upload_files = [f for f in upload_dir.iterdir() if f.is_file()]
        if upload_files:
            print(f"📁 {len(upload_files)} files found in UPLOAD")
            status['pdf_conversion_needed'] = True
        else:
            print("✅ UPLOAD directory is empty")
    
    if input_dir.exists():
        pdf_files = list(input_dir.glob("*.pdf"))
        print(f"📄 {len(pdf_files)} PDF files found in INPUT")
        
        for pdf_file in pdf_files:
            expected_md = output_dir / f"{pdf_file.stem}.md"
            expected_artifacts_dir = output_dir / f"{pdf_file.stem}_artifacts"
            
            md_exists = expected_md.exists()
            
            if md_exists:
                print(f"   ✅ {pdf_file.name} → already processed")
                status['existing_md_files'].append(expected_md)
                if expected_artifacts_dir.exists():
                    status['existing_artifacts_dirs'].append(expected_artifacts_dir)
            else:
                print(f"   🔄 {pdf_file.name} → still to process")
                status['pdf_files_to_process'].append(pdf_file)
                status['markdown_conversion_needed'] = True
    
    if output_dir.exists():
        existing_md = list(output_dir.glob("*.md"))
        existing_artifacts_dirs = [d for d in output_dir.iterdir() 
                                  if d.is_dir() and d.name.endswith('_artifacts')]
        
        print(f"📝 {len(existing_md)} Markdown files found in OUTPUT")
        print(f"🖼️ {len(existing_artifacts_dirs)} _artifacts directories found in OUTPUT")
        
        json_path = output_dir / "image_descriptions.json"
        if json_path.exists():
            print("✅ Image descriptions already available")
        else:
            print("🔄 Image descriptions missing")
    
    return status

# ========================================
# 4. IMAGE ENHANCEMENT
# ========================================

def enhance_images_contrast(directory="OUTPUT", contrast_factor=None):
    """Improves the contrast of all images in _artifacts directories."""
    print_section("4. IMAGE ENHANCEMENT")
    
    if contrast_factor is None:
        contrast_factor = CONTRAST_FACTOR
    
    enhanced_count = 0
    output_dir = Path(directory)
    artifacts_dirs = [d for d in output_dir.iterdir() 
                     if d.is_dir() and d.name.endswith('_artifacts')]
    
    if not artifacts_dirs:
        print("⚠️ No _artifacts directories found")
        return
    
    print(f"🔍 Found _artifacts directories: {len(artifacts_dirs)}")
    
    for artifacts_dir in artifacts_dirs:
        print(f"📁 Processing: {artifacts_dir.name}")
        
        for file_path in artifacts_dir.rglob("*"):
            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                try:
                    with Image.open(file_path) as img:
                        enhancer = ImageEnhance.Contrast(img)
                        enhanced_img = enhancer.enhance(contrast_factor)
                        enhanced_img.save(file_path)
                    enhanced_count += 1
                    print(f"   ✅ Enhanced: {file_path.name}")
                except Exception as e:
                    print(f"   ❌ Error with {file_path.name}: {e}")
    
    print(f"📊 Images enhanced: {enhanced_count}")

# ========================================
# 5. IMAGE DESCRIPTION WITH VLM
# ========================================

def check_api_connection():
    """Checks API connection for image descriptions."""
    try:
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 5
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat/completions", 
            json=payload, 
            headers=headers, 
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"✅ Vision API connection successful ({API_BASE_URL})")
            print(f"✅ Model '{MODEL_NAME}' is available")
            return True
        else:
            print(f"❌ Vision API not reachable (Status: {response.status_code})")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Vision API connection failed: {e}")
        return False

def query_image_with_api(image_path, retries=None):
    """Sends an Vision API request with image."""
    import base64
    
    if retries is None:
        retries: int = int(MAX_RETRIES)
        
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ Error reading image: {e}")
        return None
    
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    # LANGUAGE AGNOSTIC: Let the model respond in the same language as the document
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please describe the content of this image in great detail in the same language as the document being processed. If you detect German text or context, respond in German. If you detect English text or context, respond in English. If the language is unclear, use the language that best matches the content. Explain what can be seen, what texts or data are displayed, and what the meaning might be."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat/completions", 
                json=payload, 
                headers=headers, 
                timeout=IMAGE_DESCRIPTION_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                
                if content:
                    return content
                    
            elif response.status_code == 401:
                print(f"   ❌ Authentication failed")
                break
            else:
                print(f"   ❌ API error {response.status_code}")
                
        except requests.RequestException as e:
            print(f"   ❌ Network error (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
    
    return None



def describe_images_with_api():
    """Describes all images in _artifacts directories with VLM."""
    print_section("5. IMAGE DESCRIPTION WITH VISION API")
    
    print("🔧 Testing Vision API connection...")
    if not check_api_connection():
        print("❌ Vision API server not available")
        print("💡 Image descriptions will be skipped")
        return False
    
    output_dir = Path("OUTPUT")
    json_path = output_dir / "image_descriptions.json"
    
    existing_descriptions = {}
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    existing_descriptions[item['image_name']] = item['description']
            print(f"📄 {len(existing_descriptions)} existing descriptions loaded")
        except:
            pass
    
    image_descriptions = []
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    artifacts_dirs = [d for d in output_dir.iterdir() 
                     if d.is_dir() and d.name.endswith('_artifacts')]
    
    if not artifacts_dirs:
        print("⚠️ No _artifacts directories found")
        return False
    
    total_images = 0
    image_files = []
    for artifacts_dir in artifacts_dirs:
        for file_path in artifacts_dir.rglob("*"):
            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                relative_path = str(file_path.relative_to(output_dir))
                image_files.append((file_path, relative_path))
                total_images += 1
    
    print(f"📊 Found images: {total_images} in {len(artifacts_dirs)} directories")
    print(f"🌐 Using language-agnostic image descriptions")
    
    if total_images == 0:
        print("⚠️ No images found for description")
        return False
    
    start_time = time.time()
    
    for idx, (image_path, relative_path) in enumerate(image_files, 1):
        if relative_path in existing_descriptions:
            print(f"⏭️ Image {idx}/{total_images}: {relative_path} (already described)")
            image_descriptions.append({
                "image_name": relative_path,
                "description": existing_descriptions[relative_path]
            })
            skipped_count += 1
            continue
        
        print(f"\n🔄 Describing image {idx}/{total_images}: {relative_path}")
        
        try:
            description = query_image_with_api(image_path)
            
            if description:
                image_descriptions.append({
                    "image_name": relative_path,
                    "description": description
                })
                processed_count += 1
                print(f"   ✅ Description created ({len(description)} characters)")
            else:
                print(f"   ❌ No description received")
                failed_count += 1
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed_count += 1
            time.sleep(2)
    
    if image_descriptions:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(image_descriptions, f, ensure_ascii=False, indent=2)
        
        elapsed = time.time() - start_time
        print(f"\n📊 IMAGE DESCRIPTION COMPLETED:")
        print(f"   - Images found: {total_images}")
        print(f"   - Newly described: {processed_count}")
        print(f"   - Skipped: {skipped_count}")
        print(f"   - Failed: {failed_count}")
        print(f"   - Total time: {elapsed:.1f}s")
        if processed_count > 0:
            print(f"   - Average: {elapsed/processed_count:.1f}s per image")
        print(f"💾 Saved to: {json_path}")
        return True
    else:
        print("\n❌ No new image descriptions created")
        return False

# ========================================
# 6. MARKDOWN INTEGRATION
# ========================================

def integrate_image_descriptions():
    """Integrates image descriptions into Markdown files."""
    print_section("6. INTEGRATION OF IMAGE DESCRIPTIONS")
    
    output_dir = Path("OUTPUT")
    json_path = output_dir / "image_descriptions.json"
    
    if not json_path.exists():
        print("⚠️ No image descriptions found")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    
    print(f"📊 Loaded image descriptions: {len(descriptions)}")
    
    description_dict = {}
    for desc in descriptions:
        image_name = desc["image_name"]
        description = desc["description"]
        description_dict[image_name] = description
        
        filename_only = os.path.basename(image_name)
        description_dict[filename_only] = description
    
    md_files = [f for f in output_dir.glob("*.md") if f.name != "combined.md"]
    integrated_count = 0
    
    import re
    
    for md_file in md_files:
        print(f"🔄 Integrating into: {md_file.name}")
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = False
        
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        matches = re.findall(image_pattern, content)
        
        # Filter descriptions for this specific file to allow sequential fallback
        file_descriptions = [d for d in descriptions if f"{md_file.stem}_artifacts" in d["image_name"]]
        file_descriptions.sort(key=lambda x: x["image_name"])  # Ensure chronological order
        
        desc_index = 0
        
        for alt_text, image_path in matches:
            description = None
            
            for key in [image_path, os.path.basename(image_path)]:
                if key in description_dict:
                    description = description_dict[key]
                    break
            
            if not description:
                for desc_path in description_dict.keys():
                    if image_path in desc_path or desc_path in image_path:
                        description = description_dict[desc_path]
                        break
                        
            # Fallback to sequential index if name matching failed
            if not description and desc_index < len(file_descriptions):
                description = file_descriptions[desc_index]["description"]
            
            desc_index += 1
            
            if description:
                # Language-agnostic header - detect language from description
                if any(german_word in description.lower() for german_word in ['das', 'die', 'der', 'ist', 'sind', 'zeigt', 'abbildung']):
                    header = "**Bildbeschreibung:**"
                else:
                    header = "**Image Description:**"
                
                if f"{header} {description}" not in content:
                    old_ref = f"![{alt_text}]({image_path})"
                    new_ref = f"{old_ref}\n\n{header} {description}\n"
                    content = content.replace(old_ref, new_ref)
                    integrated_count += 1
                    changes_made = True
                    print(f"   ✅ Description integrated for: {image_path}")
        
        if changes_made:
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"   💾 File updated")
        else:
            print(f"   ⚠️ No new descriptions added")
    
    print(f"📊 Image descriptions integrated: {integrated_count}")

# ========================================
# 7. MARKDOWN COMBINATION
# ========================================

def combine_markdown_files():
    """Combines all Markdown files (optional)."""
    print_section("7. MARKDOWN COMBINATION")
    
    if not CREATE_COMBINED_MD:
        print("⚠️ Markdown combination is disabled (CREATE_COMBINED_MD = False)")
        print("💡 All Markdown files remain separate in OUTPUT/")
        return
    
    print("✅ Markdown combination is enabled")
    
    output_dir = Path("OUTPUT")
    md_files = sorted([f for f in output_dir.glob("*.md") if f.name != "combined.md"])
    
    if not md_files:
        print("⚠️ No Markdown files found to combine")
        return
    
    print(f"🔄 Combining {len(md_files)} Markdown files...")
    
    # Auto-detect primary language from first file for headers
    primary_language = "en"  # Default
    if md_files:
        try:
            with open(md_files[0], 'r', encoding='utf-8') as f:
                sample_content = f.read()[:1000].lower()
                german_indicators = ['das', 'die', 'der', 'ist', 'sind', 'und', 'mit', 'von', 'zu', 'auf']
                if any(word in sample_content for word in german_indicators):
                    primary_language = "de"
        except:
            pass
    
    # Language-specific headers
    if primary_language == "de":
        combined_content = "# Kombinierte Dokumentsammlung\n\n"
        combined_content += f"*Erstellt am: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        combined_content += f"*Aus {len(md_files)} Einzeldokumenten zusammengestellt*\n\n"
        combined_content += "## Inhaltsverzeichnis\n\n"
        source_text = "Quelle:"
    else:
        combined_content = "# Combined Document Collection\n\n"
        combined_content += f"*Created on: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        combined_content += f"*Compiled from {len(md_files)} individual documents*\n\n"
        combined_content += "## Table of Contents\n\n"
        source_text = "Source:"
    
    for i, md_file in enumerate(md_files, 1):
        combined_content += f"{i}. [{md_file.stem}](#{md_file.stem.lower().replace(' ', '-')})\n"
    combined_content += "\n---\n\n"
    
    for i, md_file in enumerate(md_files, 1):
        print(f"   📄 Adding: {md_file.name}")
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        combined_content += f"\n\n{'='*80}\n"
        combined_content += f"## {i}. {md_file.stem}\n"
        combined_content += f"{'='*80}\n\n"
        combined_content += f"*{source_text} {md_file.name}*\n\n"
        combined_content += content
        
        if i < len(md_files):
            combined_content += "\n\n---\n"
    
    combined_path = output_dir / "combined.md"
    with open(combined_path, 'w', encoding='utf-8') as f:
        f.write(combined_content)
    
    print(f"✅ Combined file created: {combined_path}")
    print(f"📊 Total length: {len(combined_content):,} characters")
    print(f"🌐 Language detected: {primary_language}")

# ========================================
# MAIN PROGRAM
# ========================================

def main():
    """Main program with optimized processing."""
    print_section("🚀 DOCUMENT PROCESSOR STARTED (WITH CENTRAL CONFIG)")
    
    print(f"🔧 CONFIGURATION:")
    print(f"   - Text extraction: Docling Standard (fast)")
    print(f"   - Image description: Vision API")
    print(f"   - Server: {API_BASE_URL}")
    print(f"   - Model: {MODEL_NAME}")
    print(f"   - Combined.md: {'YES' if CREATE_COMBINED_MD else 'NO'}")
    print(f"   - Language handling: Agnostic (preserves document language)")
    
    setup_logging()
    
    try:
        setup_directories()
        status = check_processing_status()
        
        if status['pdf_conversion_needed']:
            convert_documents_to_pdf()
            status = check_processing_status()
        
        if status['markdown_conversion_needed']:
            print_section("3. PDF TO MARKDOWN (FAST EXTRACTION)")
            if not process_remaining_pdfs(status['pdf_files_to_process']):
                print("❌ PDF conversion failed")
                return
        
        output_dir = Path("OUTPUT")
        artifacts_dirs = [d for d in output_dir.iterdir() 
                         if d.is_dir() and d.name.endswith('_artifacts')]
        
        if artifacts_dirs:
            enhance_images_contrast()
        
        json_path = output_dir / "image_descriptions.json"
        if artifacts_dirs and (not json_path.exists() or json_path.stat().st_size == 0):
            describe_images_with_api()
        
        if json_path.exists():
            integrate_image_descriptions()
        
        if CREATE_COMBINED_MD:
            combine_markdown_files()
        
        print_section("🎉 PROCESSING COMPLETED")
        
        md_files = len(list(output_dir.glob("*.md")))
        artifacts_count = len(artifacts_dirs)
        
        print(f"📊 RESULT:")
        print(f"   - Markdown files: {md_files}")
        print(f"   - _artifacts directories: {artifacts_count}")
        print(f"   - Image descriptions: {'✅' if json_path.exists() else '❌'}")
        print(f"   - Combined.md: {'✅ created' if CREATE_COMBINED_MD else '❌ disabled'}")
        print(f"   - Language preservation: ✅ Document language preserved")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logging.error(f"Main program error: {e}", exc_info=True)

if __name__ == "__main__":
    main()