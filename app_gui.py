import streamlit as st
import os
import glob
import re
from typing import Dict, List, Any, Optional, Tuple
from chat import (
    load_or_create_index, load_nlp_models, global_lda_model,
    execute_intelligent_search, list_available_topics,
    format_qa_response, format_search_response,
)

# Add enhanced mapping functionality with geocoding
try:
    import folium
    from folium.plugins import MarkerCluster, HeatMap
    from streamlit_folium import st_folium
    MAPPING_AVAILABLE = True
except ImportError:
    MAPPING_AVAILABLE = False
    st.warning("📍 Map functionality requires 'folium' and 'streamlit-folium'. Install with: pip install folium streamlit-folium")

# ⚡ NEW: Enhanced geocoding with caching
try:
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="historical_letters_app")
    geocode_cache: Dict[str, Tuple[float, float]] = {}
    GEO_CODER_AVAILABLE = True
except ImportError:
    GEO_CODER_AVAILABLE = False
    geolocator = None
    st.info("🌎 Install geopy for automatic location lookup: pip install geopy")

def initialize_session_state():
    """Initialize session state variables for the chat interface."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'letter_index' not in st.session_state:
        st.session_state.letter_index = None
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'spacy_nlp_model' not in st.session_state:
        st.session_state.spacy_nlp_model = None
    if 'sentence_model' not in st.session_state:
        st.session_state.sentence_model = None
    if 'qa_pipeline' not in st.session_state:
        st.session_state.qa_pipeline = None
    if 'force_reindex' not in st.session_state:
        st.session_state.force_reindex = False
    if 'current_action' not in st.session_state:
        st.session_state.current_action = None
    if 'use_gemini' not in st.session_state:
        st.session_state.use_gemini = True
    if 'gemini_configured' not in st.session_state:
        st.session_state.gemini_configured = False

def load_models_and_index():
    """Load all required models and document index."""
    if st.session_state.models_loaded:
        return
    
    force_reindex = st.session_state.get('force_reindex', False)
    
    if force_reindex:
        with st.spinner("🔄 Rebuilding index with 2000 letters - this may take a few minutes..."):
            _load_models_and_create_index(force_reindex=True)
    else:
        with st.spinner("🤖 Initializing AI models and loading historical documents..."):
            _load_models_and_create_index(force_reindex=False)

def _load_models_and_create_index(force_reindex=False):
    """Internal function to load models and create index."""
    # Load NLP models
    spacy_nlp_model, sentence_model, qa_pipeline = load_nlp_models()
    st.session_state.spacy_nlp_model = spacy_nlp_model
    st.session_state.sentence_model = sentence_model
    st.session_state.qa_pipeline = qa_pipeline
    
    # Get XML files
    xml_directory = "renamed_highlighted_cwrgm_xml"
    all_xml_files = glob.glob(os.path.join(xml_directory, "*.xml"))
    
    if not all_xml_files:
        st.error(f"No XML files found in {xml_directory}")
        return
    
    # Limit to 2000 files as configured
    from chat import config
    max_files = config.MAX_FILES_FOR_TESTING
    xml_files = all_xml_files[:max_files] if max_files else all_xml_files
    
    # Show file count info
    if force_reindex:
        st.info(f"📁 Found {len(all_xml_files)} XML files - processing {len(xml_files)} letters (limited to {max_files})")
    else:
        st.info(f"📁 Processing {len(xml_files)} of {len(all_xml_files)} available files")
    
    # Load document index
    index_path = "letter_index.pkl"
    
    if force_reindex:
        st.session_state.force_reindex = False  # Reset flag
    
    st.session_state.letter_index = load_or_create_index(
        xml_files, sentence_model, spacy_nlp_model, index_path, force_reindex=force_reindex
    )
    
    st.session_state.models_loaded = True
    final_count = len(st.session_state.letter_index)
    
    if force_reindex:
        st.success(f"🎉 Successfully rebuilt index with {final_count} historical documents!")
        if final_count < 2000:
            st.warning(f"⚠️ Only {final_count} letters were successfully processed (some files may have parsing issues)")
    else:
        st.success(f"✅ Loaded {final_count} historical documents and AI models!")

def display_chat_message(role, content, search_info=None):
    """Display a chat message with proper formatting."""
    with st.chat_message(role):
        st.markdown(content)
        
        if search_info and role == "assistant":
            with st.expander("ℹ️ How I found this", expanded=False):
                # Make it more conversational and less technical
                explanation = search_info.get('explanation', 'I searched through the historical letters')
                confidence = search_info.get('confidence', 0)
                
                if confidence > 0.8:
                    confidence_text = "Very confident in this result"
                elif confidence > 0.6:
                    confidence_text = "Quite confident in this result"
                elif confidence > 0.4:
                    confidence_text = "Moderately confident in this result"
                else:
                    confidence_text = "Found some relevant information"
                
                st.write(f"**How I searched:** {explanation}")
                st.write(f"**Reliability:** {confidence_text}")
                if 'error' in search_info:
                    st.error(f"Issue encountered: {search_info['error']}")

        # Show RAG citations if present
        if role == "assistant" and search_info and search_info.get('gemini_enhanced') and 'citations' in search_info:
            with st.expander("📚 Sources", expanded=False):
                for cit in search_info['citations']:
                    st.markdown(f"**{cit['ref']}** – {cit['title']} ({cit['year']}), {cit['sender']}\n\n> {cit['snippet']}")

def create_letters_map(letters_data: List[Dict]) -> Optional[folium.Map]:
    """Create an enhanced interactive map showing letter locations with clustering and better tiles."""
    if not MAPPING_AVAILABLE or not letters_data:
        return None
    
    # Create base map with better tiles - using CartoDB Positron for crisp, research-friendly appearance
    m = folium.Map(
        location=[39.0, -98.0],  # Center of United States
        zoom_start=4,
        tiles=None  # We'll add custom tiles
    )
    
    # Add multiple tile layer options for better visuals
    # CartoDB Positron - Clean, light, research-friendly
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name='Light Theme (Default)',
        overlay=False,
        control=True,
        max_zoom=19
    ).add_to(m)
    
    # CartoDB Dark Matter - Dark theme for contrast
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name='Dark Theme',
        overlay=False,
        control=True,
        max_zoom=19
    ).add_to(m)
    
    # Terrain for geographical context
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        name='Terrain',
        overlay=False,
        control=True,
        max_zoom=18
    ).add_to(m)
    
    # Historical location mappings for Civil War era
    location_coords = get_historical_location_coordinates()
    
    # Track locations and counts
    location_counts = {}
    letters_by_location = {}
    heat_data = []  # For heatmap
    
    for letter in letters_data:
        coords = None
        
        # ⚡ 1. NEW: Prefer coordinates parsed directly from XML <geo> tags
        if isinstance(letter.get("latitude"), (int, float)) and isinstance(letter.get("longitude"), (int, float)):
            coords = (letter["latitude"], letter["longitude"])
        
        # 2. Fallback to place names and historical lookup
        if coords is None:
            places = letter.get('places', [])
            if places:
                primary_place = places[0].strip()
                coords = find_coordinates_for_place(primary_place, location_coords)
                
                # ⚡ 3. NEW: Last-resort live geocoding (cached for performance)
                if coords is None and GEO_CODER_AVAILABLE and primary_place:
                    if primary_place in geocode_cache:
                        coords = geocode_cache[primary_place]
                    else:
                        try:
                            if geolocator:  # Type guard
                                location = geolocator.geocode(primary_place)
                                if location and hasattr(location, 'latitude') and hasattr(location, 'longitude'):
                                    coords = (location.latitude, location.longitude)  # type: ignore
                                    geocode_cache[primary_place] = coords
                        except Exception:
                            pass  # Geocoding failed, continue without coordinates
        
        # Only process letters with valid coordinates
        if coords:
            location_key = f"{coords[0]:.3f},{coords[1]:.3f}"
            if location_key not in location_counts:
                location_counts[location_key] = 0
                letters_by_location[location_key] = []
            
            location_counts[location_key] += 1
            letters_by_location[location_key].append(letter)
            
            # Add to heatmap data
            heat_data.append([coords[0], coords[1], 1])
    
    # Create marker cluster for better performance and visual appeal
    marker_cluster = MarkerCluster(
        name='Letter Locations (Clustered)',
        overlay=True,
        control=True,
        icon_create_function="""
        function(cluster) {
            var count = cluster.getChildCount();
            var color = count < 3 ? '#3388ff' : count < 6 ? '#ff9933' : '#ff3333';
            return L.divIcon({
                html: '<div style="background-color: ' + color + '; color: white; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; font-weight: bold; border: 2px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.3);">' + count + '</div>',
                className: 'custom-cluster-icon',
                iconSize: [40, 40]
            });
        }
        """
    ).add_to(m)
    
    # Add individual markers to cluster
    for location_key, count in location_counts.items():
        lat, lon = map(float, location_key.split(','))
        letters_at_location = letters_by_location[location_key]
        
        # Create popup content with enhanced styling
        popup_content = create_enhanced_map_popup_content(letters_at_location, count)
        
        # Determine marker color and icon based on count and content
        if count == 1:
            color = 'blue'
            icon = 'envelope'
        elif count <= 3:
            color = 'green'
            icon = 'envelopes'
        elif count <= 5:
            color = 'orange'
            icon = 'archive'
        else:
            color = 'red'
            icon = 'building'
        
        # Enhanced marker with better styling
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup_content, max_width=350, min_width=250),
            tooltip=f"📍 {count} letter(s) • Click for details",
            icon=folium.Icon(
                color=color, 
                icon=icon, 
                prefix='fa',
                icon_color='white'
            )
        ).add_to(marker_cluster)
    
    # Add heatmap layer if we have enough data points
    if len(heat_data) >= 3:
        HeatMap(
            heat_data, 
            name='Letter Density Heatmap',
            overlay=True,
            control=True,
            radius=25,
            blur=15,
            min_opacity=0.4,
            max_zoom=18,
            gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
        ).add_to(m)
    
    # Add layer control to switch between views
    folium.LayerControl(collapsed=False).add_to(m)
    
    return m

def get_historical_location_coordinates() -> Dict[str, Tuple[float, float]]:
    """Get coordinates for historical Civil War era locations."""
    return {
        # Major cities
        "Washington": (38.9072, -77.0369),
        "Washington D.C.": (38.9072, -77.0369),
        "New York": (40.7128, -74.0060),
        "Philadelphia": (39.9526, -75.1652),
        "Boston": (42.3601, -71.0589),
        "Richmond": (37.5407, -77.4360),
        "Atlanta": (33.7490, -84.3880),
        "Charleston": (32.7767, -79.9311),
        "Savannah": (32.0835, -81.0998),
        "New Orleans": (29.9511, -90.0715),
        
        # Mississippi locations
        "Jackson": (32.2988, -90.1848),
        "Vicksburg": (32.3526, -90.8779),
        "Natchez": (31.5604, -91.4031),
        "Meridian": (32.3643, -88.7037),
        "Columbus": (33.4957, -88.4277),
        "Tupelo": (34.2576, -88.7034),
        "Greenville": (33.4104, -91.0612),
        "Hattiesburg": (31.3271, -89.2903),
        "Biloxi": (30.3960, -88.8853),
        "Gulfport": (30.3674, -89.0928),
        
        # States (using capital cities)
        "Mississippi": (32.2988, -90.1848),
        "Alabama": (32.3617, -86.2792),
        "Georgia": (33.7490, -84.3880),
        "South Carolina": (34.0007, -81.0348),
        "North Carolina": (35.7796, -78.6382),
        "Virginia": (37.4316, -78.6569),
        "Tennessee": (36.1627, -86.7816),
        "Kentucky": (38.2009, -84.8733),
        "Louisiana": (30.9843, -91.9623),
        "Arkansas": (34.7465, -92.2896),
        "Texas": (30.2672, -97.7431),
        "Florida": (30.4518, -84.2807),
        
        # Military camps and battlefields
        "Gettysburg": (39.8309, -77.2311),
        "Bull Run": (38.8127, -77.5211),
        "Antietam": (39.4759, -77.7453),
        "Shiloh": (35.1495, -88.3267),
        "Chickamauga": (34.9342, -85.2547),
        "Fredericksburg": (38.3032, -77.4605),
        "Chancellorsville": (38.3099, -77.6394),
        "Cold Harbor": (37.5943, -77.1608),
        "Petersburg": (37.2279, -77.4019),
        "Appomattox": (37.3760, -78.7967),
    }

def find_coordinates_for_place(place_name: str, location_coords: Dict[str, Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Find coordinates for a place name with fuzzy matching."""
    if not place_name:
        return None
    
    place_name = place_name.strip()
    
    # Exact match first
    if place_name in location_coords:
        return location_coords[place_name]
    
    # Try case-insensitive match
    for known_place, coords in location_coords.items():
        if known_place.lower() == place_name.lower():
            return coords
    
    # Try partial matching
    for known_place, coords in location_coords.items():
        if place_name.lower() in known_place.lower() or known_place.lower() in place_name.lower():
            return coords
    
    return None

def create_enhanced_map_popup_content(letters: List[Dict], count: int) -> str:
    """Create enhanced HTML content for map popup with better styling."""
    html_content = f"""
    <div style="font-family: Arial, sans-serif; max-width: 300px;">
        <h4 style="margin: 0 0 10px 0; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px;">
            📍 {count} Letter(s) from this Location
        </h4>
    """
    
    # Show up to 4 letters in detail with enhanced styling
    for i, letter in enumerate(letters[:4]):
        title = letter.get('title', 'Untitled')
        if len(title) > 45:
            title = title[:45] + "..."
        sender = letter.get('sender', 'Unknown')
        recipient = letter.get('recipient', 'Unknown') 
        year = letter.get('year', 'Unknown')
        
        # Color coding by year for historical context
        if year != 'Unknown':
            try:
                year_int = int(year)
                if year_int <= 1861:
                    year_color = '#8e44ad'  # Purple for pre-war
                elif year_int <= 1863:
                    year_color = '#e74c3c'  # Red for early war
                elif year_int <= 1865:
                    year_color = '#f39c12'  # Orange for late war
                else:
                    year_color = '#27ae60'  # Green for post-war
            except:
                year_color = '#95a5a6'  # Gray for unknown
        else:
            year_color = '#95a5a6'
        
        html_content += f"""
        <div style="margin-bottom: 12px; padding: 8px; border-left: 4px solid {year_color}; background-color: #f8f9fa; border-radius: 0 5px 5px 0;">
            <div style="font-weight: bold; color: #2c3e50; margin-bottom: 3px;">{title}</div>
            <div style="font-size: 12px; color: #7f8c8d;">
                <span style="margin-right: 10px;">📤 <strong>From:</strong> {sender}</span><br>
                <span style="margin-right: 10px;">📥 <strong>To:</strong> {recipient}</span><br>
                <span style="background-color: {year_color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px;">
                    📅 {year}
                </span>
            </div>
        </div>
        """
    
    if count > 4:
        html_content += f"""
        <div style="text-align: center; font-style: italic; color: #7f8c8d; margin-top: 10px; padding: 5px; background-color: #ecf0f1; border-radius: 5px;">
            <i class="fa fa-plus-circle"></i> ...and {count - 4} more letter(s)
        </div>
        """
    
    html_content += "</div>"
    return html_content

def handle_button_action(action_type: str, doc: Dict, message_id: str, index: int):
    """Handle button actions by adding them to chat history naturally."""
    # Set flag to indicate button action occurred
    st.session_state.button_action_occurred = True
    
    if action_type == "full_letter":
        # Add request to chat history
        question = f"Please show me the full text of the letter: {doc.get('title', 'Unknown')} from {doc.get('sender', 'Unknown')} ({doc.get('year', 'Unknown')})"
        st.session_state.chat_history.append({
            "role": "user",
            "content": f"📖 Show full letter: {doc.get('title', 'Unknown')}"
        })
        
        # Add full letter response
        full_text = doc.get('full_text', 'Full text not available')
        word_count = len(full_text.split()) if full_text != 'Full text not available' else 0
        title = doc.get('title', 'this letter')
        sender = doc.get('sender', 'Unknown')
        recipient = doc.get('recipient', 'Unknown')
        year = doc.get('year', 'Unknown')
        
        if full_text == 'Full text not available':
            response = f"I'm sorry, but the full text of **{title}** isn't available in my database right now. "
            response += f"However, I can tell you that this letter was"
            if sender != 'Unknown':
                response += f" written by {sender}"
            if recipient != 'Unknown':
                response += f" to {recipient}"
            if year != 'Unknown':
                response += f" in {year}"
            response += "."
        else:
            response = f"Here's the complete text of **{title}**"
            if sender != 'Unknown':
                response += f", written by {sender}"
            if recipient != 'Unknown':
                response += f" to {recipient}"
            if year != 'Unknown':
                response += f" in {year}"
            response += f":\n\n**Full Letter Text ({word_count} words):**\n\n---\n\n{full_text}\n\n---"
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "search_info": {"explanation": "Full letter display", "confidence": 1.0}
        })
        
    elif action_type == "find_similar":
        # Add similar search request to chat
        question = f"Find letters similar to: {doc.get('title', 'Unknown')} from {doc.get('sender', 'Unknown')}"
        st.session_state.chat_history.append({
            "role": "user",
            "content": f"🔍 Find similar letters to: {doc.get('title', 'Unknown')}"
        })
        
        # Execute similar search
        if st.session_state.sentence_model and st.session_state.letter_index:
            from chat import execute_semantic_search
            similar_docs = execute_semantic_search(
                st.session_state.letter_index,
                doc.get('description', doc.get('title', '')),
                st.session_state.sentence_model,
                top_n=5
            )
            
            # Filter out the original document
            similar_docs = [d for d in similar_docs if d.get('file_name') != doc.get('file_name')][:3]
            
            if similar_docs:
                doc_title = doc.get('title', 'that letter')
                response = f"I found {len(similar_docs)} letters that are similar to **{doc_title}**:\n\n"
                for i, similar_doc in enumerate(similar_docs, 1):
                    title = similar_doc.get('title', 'An untitled letter')
                    sender = similar_doc.get('sender', 'Unknown author')
                    year = similar_doc.get('year', 'Unknown year')
                    similarity = similar_doc.get('similarity_score', 0)
                    
                    response += f"**{title}** was written by {sender}"
                    if year != 'Unknown year':
                        response += f" in {year}"
                    
                    if similarity > 0.8:
                        response += " - very similar content"
                    elif similarity > 0.6:
                        response += " - quite similar content"
                    else:
                        response += " - somewhat similar content"
                    response += "\n\n"
            else:
                response = f"I couldn't find any letters that are particularly similar to **{doc.get('title', 'that letter')}**. This might be because it discusses unique topics or uses distinctive language."
                
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "results": similar_docs if similar_docs else [],
                "result_type": "documents",
                "search_info": {"explanation": "Similarity search", "confidence": 0.8}
            })
        else:
            # Handle case when models aren't available
            response = "I'm sorry, I can't search for similar letters right now because the search models aren't loaded."
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "search_info": {"explanation": "Similarity search unavailable", "confidence": 0.0}
            })
    
    elif action_type == "ask_about":
        # Add interactive question prompt to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": f"❓ Ask questions about: {doc.get('title', 'Unknown')}"
        })
        
        suggestions = [
            f"What did {doc.get('sender', 'the author')} write about?",
            f"How did {doc.get('sender', 'the author')} describe their situation?",
            f"What was happening in {doc.get('year', 'this time period')}?",
            "What emotions are expressed in this letter?",
            "What specific events are mentioned?",
            "Who else is mentioned in this letter?"
        ]
        
        response = f"❓ **Ask me anything about this letter!**\n\n**Suggested questions:**\n"
        for i, suggestion in enumerate(suggestions, 1):
            response += f"{i}. {suggestion}\n"
        
        response += f"\n**Or ask your own question about this letter from {doc.get('sender', 'Unknown')}.**"
        
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": response,
            "search_info": {"explanation": "Question suggestions for specific letter", "confidence": 1.0},
            "focused_document": doc
        })

def display_document_results(results, result_type="documents", message_id=None):
    """Display search results with enhanced viewing options."""
    if not results:
        return
    
    # Create unique ID for this message's results
    if message_id is None:
        import time
        message_id = f"msg_{int(time.time() * 1000)}"
    
    if result_type == "qa":
        st.subheader(f"📋 Found {len(results)} Answer(s)")
        for i, answer in enumerate(results, 1):
            # Enhanced QA result display
            confidence_color = "🟢" if answer['confidence'] > 0.7 else "🟡" if answer['confidence'] > 0.4 else "🔴"
            
            with st.expander(f"{confidence_color} Answer {i} - {answer.get('document_title', 'Unknown')} (Confidence: {answer['confidence']:.1%})"):
                # Answer with highlighting
                st.markdown(f"**💡 Answer:** {answer['answer']}")
                
                # Show answer-specific metadata in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    if answer.get('document_sender'):
                        st.write(f"**👤 From:** {answer['document_sender']}")
                with col2:
                    if answer.get('document_year'):
                        st.write(f"**📅 Year:** {answer['document_year']}")
                with col3:
                    if answer.get('document_file'):
                        st.write(f"**📄 File:** {answer['document_file']}")
                
                # Context with enhanced formatting
                if answer.get('context_snippet'):
                    st.markdown("**📖 Context:**")
                    st.markdown(f"*{answer['context_snippet']}*")
                
                # Advanced QA info
                if answer.get('is_broad_answer'):
                    st.info("🧠 This answer comes from advanced conceptual analysis")
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"📖 Read Full Letter", key=f"qa_full_{message_id}_{i}"):
                        handle_button_action("full_letter", answer, message_id, i)
                        st.rerun()
                with col2:
                    if st.button(f"🔍 Find Similar", key=f"qa_similar_{message_id}_{i}"):
                        handle_button_action("find_similar", answer, message_id, i)
                        st.rerun()
                with col3:
                    if st.button(f"❓ Ask About This", key=f"qa_ask_{message_id}_{i}"):
                        handle_button_action("ask_about", answer, message_id, i)
                        st.rerun()
                    
    else:
        st.subheader(f"📚 Letters Found:")
        
        # Show map if we have location data
        letters_with_locations = [doc for doc in results if doc.get('places')]
        if letters_with_locations and MAPPING_AVAILABLE:
            st.markdown("### 🗺️ Letter Locations")
            letters_map = create_letters_map(letters_with_locations)
            if letters_map:
                # Create unique key for this map instance to avoid Streamlit duplicate key errors
                map_key = f"map_{message_id}_{len(letters_with_locations)}" if message_id else f"map_{len(letters_with_locations)}"
                st_folium(letters_map, width=700, height=400, key=map_key)
            else:
                st.info("📍 Map could not be created with available location data")
        
        for i, doc in enumerate(results, 1):
            title = doc.get('title', 'Untitled Letter')
            year = doc.get('year', 'Unknown')
            sender = doc.get('sender', 'Unknown')
            recipient = doc.get('recipient', 'Unknown')
            
            # Show documents prominently with key info visible
            st.markdown(f"### 📜 {i}. {title}")
            
            # Basic info in columns - always visible
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**From:** {sender}")
            with col2:
                st.write(f"**To:** {recipient}")
            with col3:
                st.write(f"**Year:** {year}")
            
            # Show relevant excerpt prominently if available
            if 'relevant_snippet' in doc and doc['relevant_snippet']:
                st.markdown("**Key excerpt:**")
                snippet = doc['relevant_snippet']
                if len(snippet) > 300:
                    snippet = snippet[:300] + "..."
                st.info(f"*\"{snippet}\"*")
            elif doc.get('description'):
                st.markdown("**About this letter:**")
                description = doc['description']
                if len(description) > 200:
                    description = description[:200] + "..."
                st.info(description)
            
            # Action buttons - prominently displayed
            st.markdown("**What would you like to do with this letter?**")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"📖 Read Full Letter", key=f"full_{message_id}_{i}", type="primary"):
                    handle_button_action("full_letter", doc, message_id, i)
                    st.rerun()
            with col2:
                if st.button(f"🔍 Find Similar Letters", key=f"similar_{message_id}_{i}"):
                    handle_button_action("find_similar", doc, message_id, i)
                    st.rerun()
            with col3:
                if st.button(f"❓ Ask Questions About This", key=f"ask_{message_id}_{i}"):
                    handle_button_action("ask_about", doc, message_id, i)
                    st.rerun()
    
            # Additional details in expandable section
            with st.expander("📋 More details about this letter"):
                col1, col2 = st.columns(2)
                with col1:
                    if doc.get('places'):
                        places = doc['places'][:3]  # Show first 3 places
                        st.write(f"**📍 Places mentioned:** {', '.join(places)}")
                    if doc.get('file_name'):
                        st.write(f"**📄 Source file:** {doc['file_name']}")
                with col2:
                    if 'similarity_score' in doc:
                        st.write(f"**🎯 Relevance score:** {doc['similarity_score']:.1%}")
                    if doc.get('word_count'):
                        st.write(f"**📝 Length:** {doc['word_count']} words")
            
            st.markdown("---")  # Separator between letters

def main():
    st.set_page_config(
        page_title="Historical Letters AI Chatbot",
        page_icon="💌",
        layout="wide"
    )
    
    # Custom CSS for better chat appearance
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stChatMessage > div {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .chat-container {
        height: 400px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    /* Improve button styling */
    .stButton > button {
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transform: translateY(-1px);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("💌 Historical Letters AI Chatbot")
    st.markdown("Ask me anything about Civil War era letters and I'll find the information for you!")
    
    initialize_session_state()
    
    # Load models if not already loaded
    if not st.session_state.models_loaded:
        load_models_and_index()
        if not st.session_state.models_loaded:
            st.stop()
    
    # Sidebar with helpful examples
    with st.sidebar:
        st.header("💡 Try asking:")
        st.markdown("""
        **Questions:**
        - "What did soldiers write about battles?"
        - "Who wrote letters about family?"
        - "How did people describe the war?"
        
        **Find by author:**
        - "Letters from John Smith"
        - "Show me letters by Sarah"
        
        **Time-based:**
        - "Letters from 1863"
        - "What happened in 1865?"
        
        **Topics:**
        - "Military correspondence"
        - "Personal family letters"
        - "Business communications"
        
        **Keywords:**
        - "Battle"
        - "Home"
        - "Money"
        """)
        
        if st.button("🗂️ Browse Available Topics"):
            if global_lda_model:
                topics = list_available_topics(global_lda_model)
                if topics:
                    st.write("**Available Topics:**")
                    for topic_id, topic_terms in topics:
                        st.write(f"Topic {topic_id}: {topic_terms}")
        
        # System Status
        st.header("📊 System Status")
        if st.session_state.models_loaded:
            st.success("✅ All AI models loaded")
            if st.session_state.letter_index:
                current_count = len(st.session_state.letter_index)
                st.info(f"📚 {current_count} letters loaded")
                
                # Show reindex option if using less than 2000 documents
                if current_count < 2000:
                    st.warning(f"⚠️ Only {current_count}/2000 letters loaded")
                    if st.button("🔄 Rebuild Index with 2000 Letters", type="primary"):
                        st.session_state.models_loaded = False
                        st.session_state.letter_index = None
                        st.session_state.force_reindex = True
                        st.rerun()
            
            # Model status
            st.write("**Available Search Methods:**")
            st.write("🤖 Question Answering" if st.session_state.qa_pipeline else "❌ Question Answering")
            st.write("🔍 Semantic Search" if st.session_state.sentence_model else "❌ Semantic Search") 
            st.write("🧠 Smart Query Processing" if st.session_state.spacy_nlp_model else "❌ Smart Query Processing")
            st.write("📈 Topic Modeling" if global_lda_model else "❌ Topic Modeling")
        else:
            st.warning("⏳ Loading AI models...")
        
        # Gemini AI Configuration
        st.header("🤖 AI Enhancement")
        
        # Check if Gemini is available
        from chat import get_gemini_agent, initialize_gemini_agent
        gemini_agent = get_gemini_agent()
        
        if gemini_agent and gemini_agent.is_available():
            st.success("✅ Gemini AI: Enhanced conversation enabled")
            if st.checkbox("Use AI-enhanced responses", value=st.session_state.use_gemini):
                st.session_state.use_gemini = True
            else:
                st.session_state.use_gemini = False
        else:
            st.warning("⚠️ Gemini AI: Not configured")
            with st.expander("🔑 Configure Gemini API"):
                st.markdown("""
                **Enable AI-enhanced conversations!**
                
                Gemini will make the chatbot more conversational and human-like while keeping all the sophisticated search capabilities.
                
                1. Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
                2. Enter it below or set GEMINI_API_KEY environment variable
                """)
                
                api_key = st.text_input("Gemini API Key", type="password", help="Your API key will not be stored permanently")
                if st.button("🚀 Enable Enhanced Conversation"):
                    if api_key:
                        try:
                            agent = initialize_gemini_agent(api_key)
                            if agent.is_available():
                                st.success("✅ Gemini AI enabled! Your conversations will now be much more natural and engaging.")
                                st.session_state.gemini_configured = True
                                st.rerun()
                            else:
                                st.error("❌ Failed to initialize Gemini. Please check your API key.")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                    else:
                        st.error("Please enter your API key")
        
        # Toggle for Retrieval-Augmented answers
        st.markdown("---")
        rag_default = st.session_state.get('use_rag', False)
        st.session_state.use_rag = st.checkbox("🔗 Use RAG answers (snippets fed to Gemini)", value=rag_default, help="Uses top letter excerpts as grounded context for Gemini. Requires Gemini to be enabled.")
        
        # Advanced Options
        st.header("⚙️ Advanced Options")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("♻️ Force Rebuild All Models"):
            st.session_state.models_loaded = False
            st.session_state.letter_index = None
            st.session_state.force_reindex = True
            # Clear cached models
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        
        # Show current configuration
        with st.expander("📋 Current Configuration"):
            st.write("**Document Limit:** 2000 letters")
            st.write("**AI Models:** spaCy + SentenceTransformers + DistilBERT")
            st.write("**Search Methods:** 6 intelligent types")
            st.write("**Topic Modeling:** LDA with 35 topics")
    
    # Main chat interface (single interface, no tabs)
    st.header("💬 Chat")
    
    # Show focused document info if active
    if st.session_state.get('focused_document'):
        focused = st.session_state['focused_document']
        col1, col2 = st.columns([4, 1])
        with col1:
            st.info(f"🎯 **Focused on:** {focused['title']} by {focused['sender']} ({focused['year']})")
        with col2:
            if st.button("❌ Clear Focus"):
                del st.session_state['focused_document']
                st.rerun()
    
    # Display chat history
    for idx, message in enumerate(st.session_state.chat_history):
        display_chat_message(
            message["role"], 
            message["content"], 
            message.get("search_info")
        )
        
        # Display results if available
        if message.get("results"):
            message_id = f"hist_{idx}"
            display_document_results(message["results"], message.get("result_type", "documents"), message_id)
    
    # Single chat input (no duplicate inputs)
    if user_input := st.chat_input("Ask me about the historical letters..."):
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        display_chat_message("user", user_input)
        
        # Modify query if focused on specific document
        search_query = user_input
        if st.session_state.get('focused_document'):
            focused = st.session_state['focused_document']
            search_query = f"{user_input} (Focus on letter from {focused['sender']} in {focused['year']})"
        
        # Process query with intelligent search
        with st.spinner("🔍 Analyzing your query and searching through historical documents..."):
            # Pass conversation history for Gemini context
            conversation_history = st.session_state.chat_history[-6:] if len(st.session_state.chat_history) > 0 else None
            
            search_result = execute_intelligent_search(
                st.session_state.letter_index,
                search_query,
                st.session_state.spacy_nlp_model,
                st.session_state.sentence_model,
                st.session_state.qa_pipeline,
                use_gemini=st.session_state.get('use_gemini', True),
                conversation_history=conversation_history,
                use_rag=st.session_state.get('use_rag', False)
            )
            
            # Filter results if focused on specific document
            if st.session_state.get('focused_document') and search_result.get("results"):
                focused_file = st.session_state['focused_document']['file_name']
                results = search_result["results"]
                focused_results = [r for r in results if r.get('file_name') == focused_file]
                other_results = [r for r in results if r.get('file_name') != focused_file]
                search_result["results"] = focused_results + other_results[:2]  # Focused first + top 2 others
            
            # Show a more natural processing indicator
            search_method = search_result["search_info"]["search_type"]
            gemini_enhanced = search_result["search_info"].get("gemini_enhanced", False)
            
            if gemini_enhanced:
                enhancement_text = " (AI-enhanced response)"
            else:
                enhancement_text = ""
                
            if search_method == "qa":
                st.success(f"🤖 Found specific answers in the letters{enhancement_text}")
            elif search_method == "sender":
                st.success(f"👤 Found letters from that person{enhancement_text}")
            elif search_method == "year":
                st.success(f"📅 Found letters from that time period{enhancement_text}")
            elif search_method == "topic":
                st.success(f"📈 Found letters discussing that topic{enhancement_text}")
            else:
                st.success(f"🔍 Found relevant historical letters{enhancement_text}")
        
        # Add assistant response to history
        assistant_message = {
            "role": "assistant",
            "content": search_result["formatted_response"],
            "search_info": search_result["search_info"],
            "results": search_result["results"],
            "result_type": search_result["result_type"]
        }
        st.session_state.chat_history.append(assistant_message)
        
        # Display assistant response
        display_chat_message(
            "assistant", 
            search_result["formatted_response"],
            search_result["search_info"]
        )
        
        # Display results with map integration
        display_document_results(
            search_result["results"], 
            search_result["result_type"],
            f"current_{len(st.session_state.chat_history)}"
        )
        
        # Auto-scroll by rerunning
        st.rerun()

if __name__ == "__main__":
    main()