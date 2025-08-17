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
# ZENTRALE KONFIGURATION LADEN
# ========================================
sys.path.append(str(Path(__file__).parent.parent.parent))  # Zum Hauptverzeichnis
from config_loader import PipelineConfigLoader

# Lade Konfiguration für dieses Modul
config_loader = PipelineConfigLoader("01_convert")
config = config_loader.get_api_config()
pipeline_config = config_loader.get_pipeline_config()

# Extrahiere Konfigurationswerte
CREATE_COMBINED_MD = config.get("create_combined_md", False)
USE_OPENAI_API = config.get("use_openai_api", True)
OPENAI_BASE_URL = config.get("openai_base_url", "http://localhost:11434/v1")
OPENAI_API_KEY = config.get("openai_api_key", "ollama")
OPENAI_MODEL_NAME = config.get("openai_model_name", "gemma3:12b-it-qat")
OLLAMA_SERVER_URL = config.get("ollama_server_url", "http://localhost:11434")
OLLAMA_API_KEY = config.get("ollama_api_key", "ollama")
OLLAMA_MODEL_NAME = config.get("ollama_model_name", "gemma3:12b-it-qat")
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_SERVER_URL}/api/chat"
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_SERVER_URL}/api/tags"
CONTRAST_FACTOR = config.get("contrast_factor", 2.0)
IMAGE_DESCRIPTION_TIMEOUT = config.get("image_description_timeout", 60)
MAX_RETRIES = config.get("max_retries", 3)

# Zeige geladene Konfiguration
print("=" * 60)
print("📋 KONFIGURATION GELADEN (01_convert)")
print("=" * 60)
config_loader.print_config_summary()
print(f"  📄 Combined MD: {'✅' if CREATE_COMBINED_MD else '❌'}")
print(f"  🖼️ Kontrast-Faktor: {CONTRAST_FACTOR}")
print("=" * 60)

# Docling-Imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

# ========================================
# HILFSFUNKTIONEN
# ========================================

def setup_logging():
    """Konfiguriert das Logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def print_section(title):
    """Druckt einen Abschnitt-Header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def get_pdf_page_count(pdf_path):
    """Ermittelt die Seitenzahl eines PDFs."""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            return len(pdf.pages)
    except:
        return 0

# ========================================
# 0. ORDNER-SETUP
# ========================================

def setup_directories():
    """Erstellt alle notwendigen Ordner."""
    directories = ['UPLOAD', 'INPUT', 'OUTPUT']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Ordner erstellt/überprüft: {directory}")

# ========================================
# 1. DOKUMENTKONVERTIERUNG ZU PDF
# ========================================

def convert_with_libreoffice(file_path, output_dir):
    """Konvertiert Dateien mit LibreOffice in PDF."""
    try:
        result = os.system(f'soffice --headless --convert-to pdf --outdir {output_dir} "{file_path}"')
        if result == 0:
            print(f"✅ LibreOffice-Konvertierung erfolgreich: {file_path}")
        else:
            print(f"❌ LibreOffice-Konvertierung fehlgeschlagen: {file_path}")
    except Exception as e:
        print(f"❌ Fehler bei LibreOffice-Konvertierung von {file_path}: {e}")

def convert_txt_to_pdf(file_path, output_path):
    """Konvertiert TXT-Dateien in PDF."""
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
        print(f"✅ TXT zu PDF konvertiert: {output_path}")
    except Exception as e:
        print(f"❌ Fehler bei TXT-Konvertierung von {file_path}: {e}")

def convert_html_to_pdf(file_path, output_path):
    """Konvertiert HTML-Dateien in PDF mit wkhtmltopdf."""
    try:
        result = os.system(f'wkhtmltopdf "{file_path}" "{output_path}"')
        if result == 0:
            print(f"✅ HTML zu PDF konvertiert: {output_path}")
        else:
            print(f"❌ HTML-Konvertierung fehlgeschlagen: {file_path}")
    except Exception as e:
        print(f"❌ Fehler bei HTML-Konvertierung von {file_path}: {e}")

def convert_documents_to_pdf():
    """Konvertiert alle Dokumente im UPLOAD-Ordner zu PDF."""
    print_section("1. DOKUMENTKONVERTIERUNG ZU PDF")
    
    upload_dir = Path('UPLOAD')
    input_dir = Path('INPUT')
    
    if not upload_dir.exists() or not any(upload_dir.iterdir()):
        print("⚠️ Keine Dateien im UPLOAD-Ordner gefunden.")
        return
    
    converted_count = 0
    
    for file_path in upload_dir.iterdir():
        if file_path.is_file():
            ext = file_path.suffix.lower()
            output_file = input_dir / (file_path.stem + '.pdf')
            
            print(f"🔄 Verarbeite: {file_path.name}")
            
            if ext == '.pdf':
                shutil.copy(file_path, output_file)
                print(f"✅ PDF kopiert: {output_file}")
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
                print(f"⚠️ Format {ext} wird nicht unterstützt: {file_path.name}")
    
    print(f"📊 Konvertierte Dateien: {converted_count}")

# ========================================
# 2. PDF ZU MARKDOWN (OHNE VLM!)
# ========================================

def setup_standard_converter():
    """Konfiguriert den DocumentConverter für schnelle Textextraktion OHNE VLM."""
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
    """Erstellt einen minimalen Converter als Fallback."""
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption()
        }
    )

def extract_images_from_result(result, pdf_file, output_dir):
    """Extrahiert und speichert Bilder aus dem Konvertierungsergebnis."""
    try:
        artifacts_dir = output_dir / f"{pdf_file.stem}_artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        extracted_images = 0
        
        if artifacts_dir.exists():
            existing_images = list(artifacts_dir.glob("*.png")) + list(artifacts_dir.glob("*.jpg"))
            if existing_images:
                print(f"📁 {len(existing_images)} Bilder bereits im Artifacts-Ordner")
                return len(existing_images)
        
        if hasattr(result.document, 'pictures'):
            for idx, picture in enumerate(result.document.pictures):
                try:
                    image_filename = artifacts_dir / f"image_{idx:03d}.png"
                    with open(image_filename, 'wb') as f:
                        f.write(picture.data)
                    extracted_images += 1
                    print(f"   🖼️ Bild extrahiert: {image_filename.name}")
                except Exception as e:
                    print(f"   ⚠️ Fehler beim Extrahieren von Bild {idx}: {e}")
        
        return extracted_images
        
    except Exception as e:
        print(f"❌ Fehler bei Bildextraktion: {e}")
        return 0

def process_remaining_pdfs(pdf_files_to_process):
    """Verarbeitet PDF-Dateien mit optimierter Standard-Konvertierung."""
    if not pdf_files_to_process:
        print("✅ Alle PDFs bereits verarbeitet")
        return True
    
    print(f"🔄 Verarbeite {len(pdf_files_to_process)} ausstehende PDF-Dateien...")
    
    output_dir = Path("OUTPUT")
    successful_conversions = 0
    total_start_time = time.time()
    
    for pdf_file in pdf_files_to_process:
        start_time = time.time()
        page_count = get_pdf_page_count(pdf_file)
        
        print(f"\n🔄 Konvertiere: {pdf_file.name}")
        if page_count > 0:
            print(f"📄 Seitenzahl: {page_count}")
        
        success = False
        
        try:
            print("🚀 Verwende optimierte Standard-Konvertierung...")
            doc_converter = setup_standard_converter()
            result = doc_converter.convert(pdf_file)
            
            md_filename = output_dir / f"{pdf_file.stem}.md"
            result.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)
            
            extracted_count = extract_images_from_result(result, pdf_file, output_dir)
            
            elapsed = time.time() - start_time
            print(f"✅ Konvertierung erfolgreich: {md_filename}")
            print(f"🖼️ Bilder extrahiert: {extracted_count}")
            print(f"⏱️ Zeit: {elapsed:.1f}s")
            if page_count > 0:
                print(f"📊 Performance: {elapsed/page_count:.2f}s pro Seite")
            
            successful_conversions += 1
            success = True
            
        except Exception as e:
            print(f"❌ Standard-Konvertierung fehlgeschlagen: {e}")
            
            try:
                print("🔄 Fallback: Basis-Konvertierung...")
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
                print(f"✅ Basis-Konvertierung erfolgreich: {md_filename}")
                print(f"⏱️ Zeit: {elapsed:.1f}s")
                
                successful_conversions += 1
                success = True
                
            except Exception as basic_error:
                print(f"❌ Auch Basis-Konvertierung fehlgeschlagen: {basic_error}")
                
                error_md = output_dir / f"{pdf_file.stem}_error.md"
                with open(error_md, 'w', encoding='utf-8') as f:
                    f.write(f"# {pdf_file.stem}\n\n**Konvertierung fehlgeschlagen**\n\n")
                    f.write(f"Die PDF-Datei konnte nicht verarbeitet werden.\n")
                    f.write(f"Fehler: {str(basic_error)}")
                print(f"⚠️ Fehler-Markdown erstellt: {error_md}")
    
    total_elapsed = time.time() - total_start_time
    print(f"\n📊 KONVERTIERUNG ABGESCHLOSSEN:")
    print(f"   - Erfolgreich: {successful_conversions}/{len(pdf_files_to_process)}")
    print(f"   - Gesamtzeit: {total_elapsed:.1f}s")
    print(f"   - Durchschnitt: {total_elapsed/len(pdf_files_to_process):.1f}s pro Datei")
    
    gc.collect()
    print("✅ Speicher freigegeben")
    
    return successful_conversions > 0

# ========================================
# 3. STATUS-ÜBERPRÜFUNG
# ========================================

def check_processing_status():
    """Überprüft welche Verarbeitungsschritte bereits abgeschlossen sind."""
    print_section("🔍 STATUS-ÜBERPRÜFUNG")
    
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
            print(f"📁 {len(upload_files)} Dateien in UPLOAD gefunden")
            status['pdf_conversion_needed'] = True
        else:
            print("✅ UPLOAD-Ordner ist leer")
    
    if input_dir.exists():
        pdf_files = list(input_dir.glob("*.pdf"))
        print(f"📄 {len(pdf_files)} PDF-Dateien in INPUT gefunden")
        
        for pdf_file in pdf_files:
            expected_md = output_dir / f"{pdf_file.stem}.md"
            expected_artifacts_dir = output_dir / f"{pdf_file.stem}_artifacts"
            
            md_exists = expected_md.exists()
            
            if md_exists:
                print(f"   ✅ {pdf_file.name} → bereits verarbeitet")
                status['existing_md_files'].append(expected_md)
                if expected_artifacts_dir.exists():
                    status['existing_artifacts_dirs'].append(expected_artifacts_dir)
            else:
                print(f"   🔄 {pdf_file.name} → noch zu verarbeiten")
                status['pdf_files_to_process'].append(pdf_file)
                status['markdown_conversion_needed'] = True
    
    if output_dir.exists():
        existing_md = list(output_dir.glob("*.md"))
        existing_artifacts_dirs = [d for d in output_dir.iterdir() 
                                  if d.is_dir() and d.name.endswith('_artifacts')]
        
        print(f"📝 {len(existing_md)} Markdown-Dateien in OUTPUT gefunden")
        print(f"🖼️ {len(existing_artifacts_dirs)} _artifacts Ordner in OUTPUT gefunden")
        
        json_path = output_dir / "image_descriptions.json"
        if json_path.exists():
            print("✅ Bildbeschreibungen bereits vorhanden")
        else:
            print("🔄 Bildbeschreibungen fehlen")
    
    return status

# ========================================
# 4. BILDVERBESSERUNG
# ========================================

def enhance_images_contrast(directory="OUTPUT", contrast_factor=None):
    """Verbessert den Kontrast aller Bilder in _artifacts Ordnern."""
    print_section("4. BILDVERBESSERUNG")
    
    if contrast_factor is None:
        contrast_factor = CONTRAST_FACTOR
    
    enhanced_count = 0
    output_dir = Path(directory)
    artifacts_dirs = [d for d in output_dir.iterdir() 
                     if d.is_dir() and d.name.endswith('_artifacts')]
    
    if not artifacts_dirs:
        print("⚠️ Keine _artifacts Ordner gefunden")
        return
    
    print(f"🔍 Gefundene _artifacts Ordner: {len(artifacts_dirs)}")
    
    for artifacts_dir in artifacts_dirs:
        print(f"📁 Verarbeite: {artifacts_dir.name}")
        
        for file_path in artifacts_dir.rglob("*"):
            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                try:
                    with Image.open(file_path) as img:
                        enhancer = ImageEnhance.Contrast(img)
                        enhanced_img = enhancer.enhance(contrast_factor)
                        enhanced_img.save(file_path)
                    enhanced_count += 1
                    print(f"   ✅ Verbessert: {file_path.name}")
                except Exception as e:
                    print(f"   ❌ Fehler bei {file_path.name}: {e}")
    
    print(f"📊 Bilder verbessert: {enhanced_count}")

# ========================================
# 5. BILDBESCHREIBUNG MIT VLM
# ========================================

def check_api_connection():
    """Überprüft API-Verbindung für Bildbeschreibungen."""
    if USE_OPENAI_API:
        return check_openai_connection()
    else:
        return check_ollama_connection()

def check_openai_connection():
    """Überprüft OpenAI-API-Verbindung."""
    try:
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": OPENAI_MODEL_NAME,
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 5
        }
        
        response = requests.post(
            f"{OPENAI_BASE_URL}/chat/completions", 
            json=payload, 
            headers=headers, 
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"✅ OpenAI-API-Verbindung erfolgreich ({OPENAI_BASE_URL})")
            print(f"✅ Modell '{OPENAI_MODEL_NAME}' ist verfügbar")
            return True
        else:
            print(f"❌ OpenAI-API nicht erreichbar (Status: {response.status_code})")
            return False
            
    except requests.RequestException as e:
        print(f"❌ OpenAI-API-Verbindung fehlgeschlagen: {e}")
        return False

def check_ollama_connection():
    """Überprüft Ollama-API-Verbindung."""
    try:
        headers = {
            'Authorization': f'Bearer {OLLAMA_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(OLLAMA_TAGS_ENDPOINT, headers=headers, timeout=10)
        
        if response.status_code == 200:
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            
            print(f"✅ Ollama-Verbindung erfolgreich ({OLLAMA_SERVER_URL})")
            
            if OLLAMA_MODEL_NAME in model_names:
                print(f"✅ Modell '{OLLAMA_MODEL_NAME}' ist verfügbar")
                return True
            else:
                print(f"❌ Modell '{OLLAMA_MODEL_NAME}' nicht gefunden!")
                return False
        else:
            print(f"❌ Ollama nicht erreichbar (Status: {response.status_code})")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Ollama-Verbindung fehlgeschlagen: {e}")
        return False

def query_image_with_api(image_path, retries=None):
    """Sendet eine API-Anfrage mit Bild für Beschreibung."""
    if retries is None:
        retries = MAX_RETRIES
        
    if USE_OPENAI_API:
        return query_openai_with_image(image_path, retries)
    else:
        return query_ollama_with_image(image_path, retries)

def query_openai_with_image(image_path, retries):
    """Sendet eine OpenAI-API-Anfrage mit Bild."""
    import base64
    
    if not os.path.exists(image_path):
        print(f"❌ Bild nicht gefunden: {image_path}")
        return None
    
    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ Fehler beim Lesen des Bildes: {e}")
        return None
    
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": OPENAI_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Bitte beschreibe den Inhalt des Bildes sehr detailliert auf Deutsch. Erkläre was zu sehen ist, welche Texte oder Daten dargestellt werden und was die Bedeutung sein könnte."
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
                f"{OPENAI_BASE_URL}/chat/completions", 
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
                print(f"   ❌ Authentifizierung fehlgeschlagen")
                break
            else:
                print(f"   ❌ API-Fehler {response.status_code}")
                
        except requests.RequestException as e:
            print(f"   ❌ Netzwerk-Fehler (Versuch {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
    
    return None

def query_ollama_with_image(image_path, retries):
    """Sendet eine Ollama-API-Anfrage mit Bild."""
    import base64
    
    if not os.path.exists(image_path):
        print(f"❌ Bild nicht gefunden: {image_path}")
        return None
    
    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ Fehler beim Lesen des Bildes: {e}")
        return None
    
    headers = {
        'Authorization': f'Bearer {OLLAMA_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": "Bitte beschreibe den Inhalt des Bildes sehr detailliert auf Deutsch. Erkläre was zu sehen ist, welche Texte oder Daten dargestellt werden und was die Bedeutung sein könnte.",
                "images": [image_base64]
            }
        ],
        "stream": False
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(
                OLLAMA_CHAT_ENDPOINT, 
                json=payload, 
                headers=headers, 
                timeout=IMAGE_DESCRIPTION_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "").strip()
                
                if content:
                    return content
                    
        except requests.RequestException as e:
            print(f"   ❌ Netzwerk-Fehler (Versuch {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
    
    return None

def describe_images_with_api():
    """Beschreibt alle Bilder in _artifacts Ordnern mit VLM."""
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    print_section(f"5. BILDBESCHREIBUNG MIT {api_name.upper()}")
    
    print(f"🔧 Teste {api_name}-Verbindung...")
    if not check_api_connection():
        print(f"❌ {api_name}-Server nicht verfügbar")
        print("💡 Bildbeschreibungen werden übersprungen")
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
            print(f"📄 {len(existing_descriptions)} existierende Beschreibungen geladen")
        except:
            pass
    
    image_descriptions = []
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    artifacts_dirs = [d for d in output_dir.iterdir() 
                     if d.is_dir() and d.name.endswith('_artifacts')]
    
    if not artifacts_dirs:
        print("⚠️ Keine _artifacts Ordner gefunden")
        return False
    
    total_images = 0
    image_files = []
    for artifacts_dir in artifacts_dirs:
        for file_path in artifacts_dir.rglob("*"):
            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                relative_path = str(file_path.relative_to(output_dir))
                image_files.append((file_path, relative_path))
                total_images += 1
    
    print(f"📊 Gefundene Bilder: {total_images} in {len(artifacts_dirs)} Ordnern")
    
    if total_images == 0:
        print("⚠️ Keine Bilder zur Beschreibung gefunden")
        return False
    
    start_time = time.time()
    
    for idx, (image_path, relative_path) in enumerate(image_files, 1):
        if relative_path in existing_descriptions:
            print(f"⏭️ Bild {idx}/{total_images}: {relative_path} (bereits beschrieben)")
            image_descriptions.append({
                "image_name": relative_path,
                "description": existing_descriptions[relative_path]
            })
            skipped_count += 1
            continue
        
        print(f"\n🔄 Beschreibe Bild {idx}/{total_images}: {relative_path}")
        
        try:
            description = query_image_with_api(image_path)
            
            if description:
                image_descriptions.append({
                    "image_name": relative_path,
                    "description": description
                })
                processed_count += 1
                print(f"   ✅ Beschreibung erstellt ({len(description)} Zeichen)")
            else:
                print(f"   ❌ Keine Beschreibung erhalten")
                failed_count += 1
            
        except Exception as e:
            print(f"   ❌ Fehler: {e}")
            failed_count += 1
            time.sleep(2)
    
    if image_descriptions:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(image_descriptions, f, ensure_ascii=False, indent=2)
        
        elapsed = time.time() - start_time
        print(f"\n📊 BILDBESCHREIBUNG ABGESCHLOSSEN:")
        print(f"   - Bilder gefunden: {total_images}")
        print(f"   - Neu beschrieben: {processed_count}")
        print(f"   - Übersprungen: {skipped_count}")
        print(f"   - Fehlgeschlagen: {failed_count}")
        print(f"   - Gesamtzeit: {elapsed:.1f}s")
        if processed_count > 0:
            print(f"   - Durchschnitt: {elapsed/processed_count:.1f}s pro Bild")
        print(f"💾 Gespeichert in: {json_path}")
        return True
    else:
        print("\n❌ Keine neuen Bildbeschreibungen erstellt")
        return False

# ========================================
# 6. MARKDOWN-INTEGRATION
# ========================================

def integrate_image_descriptions():
    """Integriert Bildbeschreibungen in Markdown-Dateien."""
    print_section("6. INTEGRATION VON BILDBESCHREIBUNGEN")
    
    output_dir = Path("OUTPUT")
    json_path = output_dir / "image_descriptions.json"
    
    if not json_path.exists():
        print("⚠️ Keine Bildbeschreibungen gefunden")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    
    print(f"📊 Geladene Bildbeschreibungen: {len(descriptions)}")
    
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
        print(f"🔄 Integriere in: {md_file.name}")
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = False
        
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        matches = re.findall(image_pattern, content)
        
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
            
            if description:
                if f"**Bildbeschreibung:** {description}" not in content:
                    old_ref = f"![{alt_text}]({image_path})"
                    new_ref = f"{old_ref}\n\n**Bildbeschreibung:** {description}\n"
                    content = content.replace(old_ref, new_ref)
                    integrated_count += 1
                    changes_made = True
                    print(f"   ✅ Beschreibung integriert für: {image_path}")
        
        if changes_made:
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"   💾 Datei aktualisiert")
        else:
            print(f"   ⚠️ Keine neuen Beschreibungen hinzugefügt")
    
    print(f"📊 Bildbeschreibungen integriert: {integrated_count}")

# ========================================
# 7. MARKDOWN-KOMBINATION
# ========================================

def combine_markdown_files():
    """Kombiniert alle Markdown-Dateien (optional)."""
    print_section("7. MARKDOWN-KOMBINATION")
    
    if not CREATE_COMBINED_MD:
        print("⚠️ Markdown-Kombination ist deaktiviert (CREATE_COMBINED_MD = False)")
        print("💡 Alle Markdown-Dateien bleiben einzeln in OUTPUT/")
        return
    
    print("✅ Markdown-Kombination ist aktiviert")
    
    output_dir = Path("OUTPUT")
    md_files = sorted([f for f in output_dir.glob("*.md") if f.name != "combined.md"])
    
    if not md_files:
        print("⚠️ Keine Markdown-Dateien zum Kombinieren gefunden")
        return
    
    print(f"🔄 Kombiniere {len(md_files)} Markdown-Dateien...")
    
    combined_content = "# Kombinierte Dokumentsammlung\n\n"
    combined_content += f"*Erstellt am: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    combined_content += f"*Aus {len(md_files)} Einzeldokumenten zusammengestellt*\n\n"
    
    combined_content += "## Inhaltsverzeichnis\n\n"
    for i, md_file in enumerate(md_files, 1):
        combined_content += f"{i}. [{md_file.stem}](#{md_file.stem.lower().replace(' ', '-')})\n"
    combined_content += "\n---\n\n"
    
    for i, md_file in enumerate(md_files, 1):
        print(f"   📄 Füge hinzu: {md_file.name}")
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        combined_content += f"\n\n{'='*80}\n"
        combined_content += f"## {i}. {md_file.stem}\n"
        combined_content += f"{'='*80}\n\n"
        combined_content += f"*Quelle: {md_file.name}*\n\n"
        combined_content += content
        
        if i < len(md_files):
            combined_content += "\n\n---\n"
    
    combined_path = output_dir / "combined.md"
    with open(combined_path, 'w', encoding='utf-8') as f:
        f.write(combined_content)
    
    print(f"✅ Kombinierte Datei erstellt: {combined_path}")
    print(f"📊 Gesamtlänge: {len(combined_content):,} Zeichen")

# ========================================
# HAUPTPROGRAMM
# ========================================

def main():
    """Hauptprogramm mit optimierter Verarbeitung."""
    print_section("🚀 DOKUMENTEN-PROCESSOR GESTARTET (MIT ZENTRALER CONFIG)")
    
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    server_url = OPENAI_BASE_URL if USE_OPENAI_API else OLLAMA_SERVER_URL
    model_name = OPENAI_MODEL_NAME if USE_OPENAI_API else OLLAMA_MODEL_NAME
    
    print(f"🔧 KONFIGURATION:")
    print(f"   - Textextraktion: Docling Standard (schnell)")
    print(f"   - Bildbeschreibung: {api_name}")
    print(f"   - Server: {server_url}")
    print(f"   - Modell: {model_name}")
    print(f"   - Combined.md: {'JA' if CREATE_COMBINED_MD else 'NEIN'}")
    
    setup_logging()
    
    try:
        setup_directories()
        status = check_processing_status()
        
        if status['pdf_conversion_needed']:
            convert_documents_to_pdf()
            status = check_processing_status()
        
        if status['markdown_conversion_needed']:
            print_section("3. PDF ZU MARKDOWN (SCHNELLE EXTRAKTION)")
            if not process_remaining_pdfs(status['pdf_files_to_process']):
                print("❌ PDF-Konvertierung fehlgeschlagen")
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
        
        print_section("🎉 VERARBEITUNG ABGESCHLOSSEN")
        
        md_files = len(list(output_dir.glob("*.md")))
        artifacts_count = len(artifacts_dirs)
        
        print(f"📊 ERGEBNIS:")
        print(f"   - Markdown-Dateien: {md_files}")
        print(f"   - _artifacts Ordner: {artifacts_count}")
        print(f"   - Bildbeschreibungen: {'✅' if json_path.exists() else '❌'}")
        print(f"   - Combined.md: {'✅ erstellt' if CREATE_COMBINED_MD else '❌ deaktiviert'}")
        
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        logging.error(f"Hauptprogramm-Fehler: {e}", exc_info=True)

if __name__ == "__main__":
    main()