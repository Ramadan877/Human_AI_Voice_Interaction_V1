from flask import Flask, request, render_template, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename
import openai
import os
from gtts import gTTS
import whisper  
import json
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
import time
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = 'uploads/'
USER_AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, 'user_audio')
AI_AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, 'ai_audio')
CONCEPT_AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, 'concept_audio')
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'txt', 'png', 'jpg', 'jpeg', 'gif', 'pdf', 'mp4', 'wav', 'mp3', 'ogg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['USER_AUDIO_FOLDER'] = USER_AUDIO_FOLDER
app.config['AI_AUDIO_FOLDER'] = AI_AUDIO_FOLDER
app.config['CONCEPT_AUDIO_FOLDER'] = CONCEPT_AUDIO_FOLDER

def check_paths():
    """Verify all required paths exist and are writable."""
    paths = [
        app.config['UPLOAD_FOLDER'],
        app.config['USER_AUDIO_FOLDER'],
        app.config['AI_AUDIO_FOLDER'],
        app.config['CONCEPT_AUDIO_FOLDER'],
        STATIC_FOLDER
    ]
    for path in paths:
        if not os.path.exists(path):
            logger.info(f"Creating path: {path}")
            os.makedirs(path, exist_ok=True)
        if not os.access(path, os.W_OK):
            logger.warning(f"WARNING: Path not writable: {path}")
            return False
    return True

def allowed_file(filename):
    """Check if the file type is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize whisper model
try:
    logger.info("Loading Whisper model...")
    model = whisper.load_model("small")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {str(e)}")
    model = None

def speech_to_text(audio_file_path):
    """Convert audio to text using OpenAI Whisper API or local fallback."""
    if not os.path.exists(audio_file_path):
        logger.error(f"Audio file does not exist: {audio_file_path}")
        return "Error: Audio file not found"
    
    logger.info(f"Processing audio file: {audio_file_path}")
    
    try:
        with open(audio_file_path, "rb") as audio_file:
            logger.info("Calling OpenAI Whisper API...")
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file
            )
        logger.info("OpenAI Whisper API transcription successful")
        return transcript.text
    except Exception as e:
        logger.error(f"Error using OpenAI Whisper API: {str(e)}")
        logger.info("Falling back to local Whisper model...")
        
        if model is not None:
            try:
                logger.info("Using local Whisper model for transcription...")
                result = model.transcribe(audio_file_path)
                logger.info("Local Whisper transcription successful")
                return result["text"]
            except Exception as e2:
                logger.error(f"Error using local Whisper model: {str(e2)}")
        else:
            logger.error("Local Whisper model not available")
            
        return "Sorry, I couldn't understand the audio."

def generate_intro_audio():
    """Generate the introductory audio message for the chatbot."""
    logger.info("Generating intro audio...")
    
    intro_text = "Hello, let us begin the self-explanation journey, go through the following concepts of uni and multivariate analysis, and try explaining what you understood from these concepts one by one in your own words!"
    intro_audio_filename = 'intro_message.mp3'
    
    intro_audio_path = os.path.join(app.config['AI_AUDIO_FOLDER'], intro_audio_filename)

    success = generate_audio(intro_text, intro_audio_path)

    if success:
        logger.info(f"Intro audio generated successfully at: {intro_audio_path}")
        return intro_audio_path
    else:
        logger.error("Failed to generate intro audio")
        return None

def generate_audio(text, file_path):
    """Generate speech (audio) from the provided text using gTTS."""
    try:
        logger.info(f"Generating audio for text: {text[:30]}...")
        tts = gTTS(text=text, lang='en')
        tts.save(file_path)
        
        if os.path.exists(file_path):
            logger.info(f"Audio file successfully saved: {file_path}")
            return True
        else:
            logger.error(f"Failed to save audio file: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return False

def save_transcript(text, source_type, sequence_id=None):
    """Save transcript text to a file with sequential naming.
    
    Args:
        text (str): The transcribed text to save
        source_type (str): Either 'user' or 'ai'
        sequence_id (int, optional): Sequence number for the transcript. If None, will be determined.
    
    Returns:
        str: Path to the saved transcript file
    """
    # Create transcript folder if it doesn't exist
    transcript_folder = r"C:\Users\Abdelrahman Ramadan\Desktop\Master Thesis\HAI\V1\uploads\Texts"
    os.makedirs(transcript_folder, exist_ok=True)
    
    # If sequence_id is not provided, determine the next sequence number
    if sequence_id is None:
        # Get existing transcript files
        existing_files = [f for f in os.listdir(transcript_folder) 
                         if f.startswith(f"{source_type}_transcript_") and f.endswith(".txt")]
        
        # Extract sequence numbers from existing files
        sequence_numbers = []
        for filename in existing_files:
            try:
                # Extract number between source_type_transcript_ and .txt
                seq_str = filename.replace(f"{source_type}_transcript_", "").replace(".txt", "")
                if seq_str.isdigit():
                    sequence_numbers.append(int(seq_str))
            except:
                pass
        
        # Determine next sequence number
        sequence_id = 1 if not sequence_numbers else max(sequence_numbers) + 1
    
    # Create filename with sequential numbering
    filename = f"{source_type}_transcript_{sequence_id}.txt"
    file_path = os.path.join(transcript_folder, filename)
    
    # Save the text
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Saved {source_type} transcript to {file_path}")
        return file_path, sequence_id
    except Exception as e:
        logger.error(f"Error saving transcript: {str(e)}")
        return None, None

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/resources/<path:filename>')
def download_resource(filename):
    """Serve resources like PDF or video files from the resources folder."""
    return send_from_directory('resources', filename)

def load_concepts():
    """Load concepts from a JSON file or create if it doesn't exist."""
    try:
        logger.info("Loading concepts from JSON file...")
        with open("concepts.json", "r") as file:
            concepts = json.load(file)["concepts"]
            logger.info(f"Loaded {len(concepts)} concepts successfully")
            return concepts
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Concepts file not found or invalid: {str(e)}. Creating default concepts...")
        concepts = [
            {
                "name": "Contract Type Analysis",
                "description": "Analysis of risk patterns between cash loans and revolving loans.",
                "golden_answer": "Cash loans have a higher default rate (>8%) compared to revolving loans (>5%). This indicates lower risk is associated with revolving loans, which should be prioritized first when considering loan approvals."
            },
            {
                "name": "Age Group Risk Analysis",
                "description": "Analysis of default rates across different age groups of clients.",
                "golden_answer": "There's a clear pattern where default rates decrease with age: 10% for ages 20-40, 7% for ages 40-60, and 5% for ages 60+. This indicates that older clients present lower risk for loan approvals, with less risk associated with people of advanced age."
            },
            {
                "name": "Income Level Risk Analysis",
                "description": "Examination of how income levels correlate with loan default rates.",
                "golden_answer": "Clients with total income above 3 Lakhs have lower default rates, while those with income below 2 Lakhs have higher default rates. The pattern shows that higher income correlates with lower default risk, making higher-income clients preferred candidates for loan approval."
            },
            {
                "name": "Income Type Risk Analysis",
                "description": "Analysis of default patterns across different income sources.",
                "golden_answer": "Pensioners and State Servants have relatively lower default rates, making them lower-risk clients. Working professionals and commercial associates have slightly higher default rates. Very high default percentages appear in 'Maternity Leave' and 'Unemployed' categories, but these can be ignored due to low sample counts."
            },
            {
                "name": "Education Type Risk Analysis",
                "description": "Examination of how education level affects default risk.",
                "golden_answer": "Clients with Academic degrees or Higher education tend to default less frequently, indicating lower risk. Clients with Lower secondary, Secondary special, or Incomplete higher education show higher default rates. This suggests education level is a useful risk indicator when evaluating loan applications."
            },
            {
                "name": "Occupation Type Risk Analysis",
                "description": "Analysis of default patterns across different occupations.",
                "golden_answer": "Lower-risk occupations include Core staff, Managers, High-skill tech staff, Accountants, Medicine professionals, Private service workers, Secretaries, HR, and IT staff. Higher-risk occupations showing elevated default rates include Sales staff, Drivers, Security staff, Cooking staff, Low-skill Laborers (very high risk), and Waiters/barmen. Other occupations fall within an acceptable range of 7.5-9% default rate."
            },
            {
                "name": "Last Information Change Risk Analysis",
                "description": "Analysis of how recently changed registration or ID information correlates with default risk.",
                "golden_answer": "Clients who changed their information within the last 5 years have higher default rates. Those who changed information more than 5 years ago have lower default rates. The lowest risk is associated with clients who changed their information more than 16 years ago, indicating stability might be a positive risk indicator."
            },
            {
                "name": "Income vs Credit Request Analysis",
                "description": "Bivariate analysis of income levels against credit amount requested.",
                "golden_answer": "Clients with lower income who request credit amounts higher than their income show high default rates (10%). Clients with higher income who request proportional credit amounts have much lower default rates (5-6%). This suggests that the ratio of requested credit to income is an important risk factor."
            },
            {
                "name": "Income vs Age Analysis",
                "description": "Bivariate analysis of income levels across different age groups.",
                "golden_answer": "Young clients (age 20-40) with lower income levels (<2-3 Lakh) show high default rates. Clients older than 50 across all income levels demonstrate lower default rates. The pattern reveals that both higher age and higher income correlate with lower default rates, with the combination of the two factors presenting the lowest risk."
            },
            {
                "name": "Income vs Housing Type Analysis",
                "description": "Bivariate analysis examining income levels against different housing situations.",
                "golden_answer": "Lower-income clients (<2 Lakhs) living in rented apartments or with parents show very high default rates (10-13%). Clients with owned apartments/houses or office apartments show average default rates (6-8%). Clients with office apartments or owned houses/apartments AND high income demonstrate very low default rates (3-6%), indicating this combination presents the lowest risk."
            },
            {
                "name": "Income vs Last Phone Change Analysis",
                "description": "Bivariate analysis of income levels against recency of phone number changes.",
                "golden_answer": "Clients with lower income (<2-3 lakhs) who changed their phone numbers recently (<2-3 years) have higher default rates. Clients who have maintained the same phone number for longer periods (>5-6 years) have lower default rates, suggesting stability in contact information may be a positive risk indicator."
            },
            {
                "name": "Income vs Region Rating Analysis",
                "description": "Bivariate analysis of income levels against the rating of client's region.",
                "golden_answer": "Clients from regions with ratings <1.5 show lower default rates across all income levels. Clients from regions rated 2-2.5 have acceptable default rates. Clients from regions rated 3 show high default rates regardless of income level, suggesting regional factors significantly impact default risk independent of income."
            },
            {
                "name": "Previous Contract Status Analysis",
                "description": "Analysis of how previous application status affects default risk.",
                "golden_answer": "Clients whose previous applications were rejected show a very high default rate (~12%). Those who previously cancelled their applications have a moderate default rate (~9%). Clients whose previous applications were approved show lower default rates (~7.5%). This suggests the bank's previous rejection decisions were largely justified based on risk factors."
            }
        ]
        
        try:
            with open("concepts.json", "w") as file:
                json.dump({"concepts": concepts}, file, indent=4)
            logger.info("Default concepts created and saved successfully")
        except Exception as write_err:
            logger.error(f"Error saving default concepts: {str(write_err)}")
        
        return concepts
        
@app.route('/set_context', methods=['POST'])
def set_context():
    """Set the context for a specific concept from the provided material."""
    concept_name = request.form.get('concept_name')
    logger.info(f"Setting context for concept: {concept_name}")
    
    concepts = load_concepts()
    
    selected_concept = next((c for c in concepts if c["name"] == concept_name), None)

    if not selected_concept:
        logger.error(f"Invalid concept selection: {concept_name}")
        return jsonify({'error': 'Invalid concept selection'})

    session['concept_name'] = selected_concept["name"]
    session['description'] = selected_concept["description"]
    session['golden_answer'] = selected_concept["golden_answer"]
    session['attempt_count'] = 0

    logger.info(f"Context set successfully for: {selected_concept['name']}")
    return jsonify({'message': f'Context set for {selected_concept["name"]}.'})

@app.route('/get_intro_audio', methods=['GET'])
def get_intro_audio():
    """Generate and serve the introductory audio message for the chatbot."""
    logger.info("Handling request for intro audio")
    
    intro_text = "Hello, let us begin the self-explanation journey, just go through each concept of the following Natural Language Processing concepts, and then go on with explaining what you understood from each concept!"
    intro_audio_filename = 'intro_message.mp3'
    
    intro_audio_path = os.path.join(app.config['AI_AUDIO_FOLDER'], intro_audio_filename)

    # Ensure directory exists
    os.makedirs(app.config['AI_AUDIO_FOLDER'], exist_ok=True)
    
    # Generate audio
    success = generate_audio(intro_text, intro_audio_path)
    
    # Initialize conversation sequence in session
    session['conversation_seq'] = 0
    
    # Save intro transcript with sequence 0
    save_transcript(intro_text, "ai_intro", 0)
    
    if success and os.path.exists(intro_audio_path):
        intro_audio_url = f"/uploads/ai_audio/{intro_audio_filename}"
        logger.info(f"Intro audio generated successfully. URL: {intro_audio_url}")
        return jsonify({'intro_audio_url': intro_audio_url})
    else:
        logger.error("Failed to generate introduction audio")
        return jsonify({'error': 'Failed to generate introduction audio'}), 500

@app.route('/get_concept_audio/<concept_name>', methods=['GET'])
def get_concept_audio(concept_name):
    """Generate concept introduction audio message."""
    logger.info(f"Generating audio for concept: {concept_name}")
    
    safe_concept = secure_filename(concept_name)
    concept_audio_filename = f'{safe_concept}_intro.mp3'
    
    concept_audio_path = os.path.join(app.config['CONCEPT_AUDIO_FOLDER'], concept_audio_filename)
    
    # Ensure directory exists
    os.makedirs(app.config['CONCEPT_AUDIO_FOLDER'], exist_ok=True)
    
    # Get current sequence number
    concept_seq = session.get('concept_seq', 0)
    concept_seq += 1
    session['concept_seq'] = concept_seq
    
    concept_intro_text = f"Now go through this concept of {concept_name}, and try explaining what you understood from this concept in your own words!"
    
    # Save concept intro transcript with sequential ID
    save_transcript(concept_intro_text, f"concept_{safe_concept}", concept_seq)
    
    success = generate_audio(concept_intro_text, concept_audio_path)
    
    if success and os.path.exists(concept_audio_path):
        logger.info(f"Concept audio generated successfully at: {concept_audio_path}")
        return send_from_directory(app.config['CONCEPT_AUDIO_FOLDER'], concept_audio_filename)
    else:
        logger.error(f"Failed to generate concept audio for: {concept_name}")
        return jsonify({'error': 'Failed to generate concept audio'}), 500
        
@app.route('/submit_message', methods=['POST'])
def submit_message():
    """Handle the submission of user messages and generate AI responses."""
    user_message = request.form.get('message')
    concept_name = request.form.get('concept_name')
    
    logger.info(f"Received message submission for concept: {concept_name}")
    logger.info(f"User message provided: {'Yes' if user_message else 'No'}")
    logger.info(f"Audio file provided: {'Yes' if 'audio' in request.files else 'No'}")

    # Process audio file if provided
    if 'audio' in request.files and request.files['audio'].filename:
        audio_file = request.files['audio']
        logger.info(f"Processing audio file: {audio_file.filename}")
        
        # Debug audio file
        logger.info(f"Audio file content type: {audio_file.content_type}")
        logger.info(f"Audio file size: {audio_file.content_length} bytes")
        
        # Create directories if they don't exist
        os.makedirs(app.config['USER_AUDIO_FOLDER'], exist_ok=True)
        
        # Save the original audio
        original_filename = secure_filename(audio_file.filename)
        original_path = os.path.join(app.config['USER_AUDIO_FOLDER'], original_filename)
        
        try:
            audio_file.save(original_path)
            logger.info(f"Original audio saved at: {original_path}")
            
            # Generate a unique filename for the converted audio
            current_timestamp = int(time.time())
            audio_path = os.path.join(app.config['USER_AUDIO_FOLDER'], f"user_audio_{current_timestamp}.wav")
            
            # Check file exists and has content
            if not os.path.exists(original_path) or os.path.getsize(original_path) == 0:
                logger.error(f"Saved audio file is empty or missing: {original_path}")
                return jsonify({'error': 'Audio file is empty or missing'})
            
            # Check if already WAV
            if original_path.lower().endswith('.wav'):
                # Just copy or use as is
                if original_path != audio_path:
                    import shutil
                    shutil.copy2(original_path, audio_path)
                    logger.info(f"Copied WAV file to: {audio_path}")
            else:
                # Convert to WAV
                try:
                    logger.info(f"Converting audio file to WAV: {original_path} -> {audio_path}")
                    # For WebM specifically, which is common from browser recording
                    if original_path.lower().endswith('.webm') or 'webm' in audio_file.content_type.lower():
                        import subprocess
                        cmd = ['ffmpeg', '-i', original_path, audio_path]
                        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        logger.info(f"FFmpeg conversion output: {process.stderr.decode()}")
                        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                            raise Exception("FFmpeg conversion failed")
                    else:
                        # Use pydub for other formats
                        sound = AudioSegment.from_file(original_path)
                        sound.export(audio_path, format="wav")
                    logger.info(f"Audio converted successfully to: {audio_path}")
                except Exception as e:
                    logger.error(f"Error converting audio: {str(e)}")
                    return jsonify({'error': f'Audio conversion failed: {str(e)}'})
            
            # Check if WAV file exists and has content
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                logger.error(f"Converted WAV file is empty or missing: {audio_path}")
                return jsonify({'error': 'Converted WAV file is empty or missing'})
                
            # Transcribe audio
            user_message = speech_to_text(audio_path)
            logger.info(f"Transcription result: {user_message}")
            
        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
            return jsonify({'error': f'Audio processing failed: {str(e)}'})

    if not user_message:
        logger.error("No message or audio transcription available")
        return jsonify({'error': 'Message or audio is required.'})

    # Get conversation sequence from session or initialize it
    conversation_seq = session.get('conversation_seq', 0)
    conversation_seq += 1
    session['conversation_seq'] = conversation_seq

    # Save user transcript with sequential ID
    user_transcript_path, _ = save_transcript(user_message, "user", conversation_seq)

    if not concept_name:
        logger.error("No concept detected in submission")
        concept_name = "Default Concept"  # Provide fallback
        logger.info(f"Using default concept: {concept_name}")

    concepts = load_concepts()
    selected_concept = next((c for c in concepts if c["name"] == concept_name), None)

    if not selected_concept:
        logger.error(f"Concept not found in system: {concept_name}")
        # Create a temporary concept for fallback
        selected_concept = {
            "name": concept_name,
            "description": f"Explaining the concept of {concept_name}",
            "golden_answer": f"A thorough understanding of {concept_name} covers its key principles and applications."
        }
        logger.info(f"Created temporary concept: {selected_concept['name']}")

    attempt_count = session.get('attempt_count', 0)
    
    ai_response = generate_response(
        user_message,
        selected_concept["name"],
        selected_concept["description"],
        selected_concept["golden_answer"],
        attempt_count
    )
    
    session['attempt_count'] = attempt_count + 1

    if not ai_response:
        logger.error("AI response generation failed")
        return jsonify({'error': 'AI response generation failed.'})

    logger.info(f"AI Response generated: {ai_response[:50]}...")

    # Save AI transcript with the same sequence ID
    ai_transcript_path, _ = save_transcript(ai_response, "ai", conversation_seq)

    # Generate audio response - still using timestamp for audio filename to avoid conflicts
    current_timestamp = int(time.time())
    ai_response_filename = f"response_audio_{current_timestamp}.mp3"
    audio_response_path = os.path.join(app.config['AI_AUDIO_FOLDER'], ai_response_filename)
    
    # Ensure directory exists
    os.makedirs(app.config['AI_AUDIO_FOLDER'], exist_ok=True)
    
    success = generate_audio(ai_response, audio_response_path)
    
    if not success or not os.path.exists(audio_response_path):
        logger.error("AI audio file not created")
        return jsonify({'error': 'AI audio generation failed.'})

    ai_audio_url = f"/uploads/ai_audio/{ai_response_filename}"
    logger.info(f"AI Response Audio URL: {ai_audio_url}")

    return jsonify({
        'response': ai_response,
        'ai_audio_url': ai_audio_url
    })

def generate_response(user_message, concept_name, concept_description, golden_answer, attempt_count):
    """Generate a response dynamically using OpenAI GPT."""
    logger.info(f"Generating response for attempt {attempt_count}")

    if not golden_answer or not concept_description:
        logger.warning("Missing context for response generation")
        return "As your tutor, I'm not able to provide you with feedback without having context about your explanation. Please ensure the context is set."
    
    base_prompt = f"""
    Concept: {concept_name}
    Context: {concept_description}
    Golden Answer: {golden_answer}
    User Explanation: {user_message}
    
    You are a friendly and encouraging tutor, helping a student refine their understanding in a supportive way. Your goal is to evaluate the student's explanation and provide warm, engaging feedback:
    - If the user's explanation is very accurate, celebrate their effort and reinforce their confidence.
    - If the explanation is partially correct, acknowledge their progress and gently guide them toward refining their answer.
    - If it's incorrect, provide constructive and positive feedback without discouraging them. Offer hints and encouragement.
    - Use a conversational tone, making the user feel comfortable and motivated to keep trying.
    - Offer increasingly specific hints or the correct answer after multiple attempts, always keeping a friendly and supportive attitude.
    """

    if attempt_count == 1:
        base_prompt += "\nProvide general feedback and a broad hint to guide the user."
    elif attempt_count == 2:
        base_prompt += "\nProvide more specific feedback and highlight key elements the user missed."
    elif attempt_count >= 3:
        base_prompt += "\nProvide the correct explanation, as the user has made multiple attempts."

    try:
        logger.info("Calling OpenAI API for response generation")
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful tutor providing feedback to students."},
                {"role": "user", "content": base_prompt}
            ],
            max_tokens=200,
            temperature=0.7,
        )
        ai_response = response['choices'][0]['message']['content']
        logger.info("OpenAI API response received successfully")
        return ai_response
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        return f"I'm having trouble generating a response right now. Please try again in a moment."

@app.route('/uploads/<folder>/<filename>')
def serve_audio(folder, filename):
    """Serve the audio files from the uploads folder."""
    logger.info(f"Serving audio from folder: {folder}, file: {filename}")
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
    
    if not os.path.exists(os.path.join(folder_path, filename)):
        logger.error(f"File not found: {os.path.join(folder_path, filename)}")
        return "File not found", 404
        
    return send_from_directory(folder_path, filename)

@app.route('/pdf')
def serve_pdf():
    return send_from_directory('resources', '1_NLP_cleaning_and_preprocessing.pdf')

if __name__ == '__main__':
    if check_paths():
        logger.info("All paths verified successfully")
        intro_path = generate_intro_audio()
        if intro_path:
            logger.info(f"Intro audio created at: {intro_path}")
        else:
            logger.warning("Failed to create intro audio")
        app.run(debug=True)
    else:
        logger.error("Path verification failed. Check permissions and try again.")


























