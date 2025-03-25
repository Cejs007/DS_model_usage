import streamlit as st
import os
import matplotlib.pyplot as plt
from model import ImageClassifier

def plot_predictions(predictions):
    """Vytvoří bar plot pro top 3 predikce"""
    plt.figure(figsize=(10, 5))
    classes = [pred[1] for pred in predictions]
    scores = [pred[2] * 100 for pred in predictions]
    
    plt.barh(classes, scores)
    plt.xlabel('Pravděpodobnost (%)')
    plt.title('Top 3 predikované třídy')
    plt.tight_layout()
    return plt

def main():
    st.set_page_config(layout="wide")
    
    # Inicializace klasifikátoru
    classifier = ImageClassifier()
    
    st.title("🖼️ Klasifikace obrázků pomocí MobileNetV2")
    
    # Sloupce pro lepší rozložení
    col1, col2 = st.columns(2)
    
    with col1:
        # Upload obrázku
        uploaded_file = st.file_uploader(
            "Nahrajte obrázek", 
            type=["jpg", "jpeg", "png"],
            help="Podporované formáty: JPG, PNG"
        )
    
    if uploaded_file is not None:
        # Uložení nahraného souboru
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with col1:
            # Zobrazení nahraného obrázku
            st.image(uploaded_file, caption="Nahraný obrázek", use_column_width=True)
        
        with col2:
            st.subheader("Výsledky klasifikace")
            
            try:
                # Predikce
                predictions = classifier.predict(temp_file)
                
                # Zobrazení výsledků
                for i, (idx, label, score) in enumerate(predictions, 1):
                    st.metric(
                        label=f"{i}. {label}", 
                        value=f"{score*100:.2f}%"
                    )
                
                # Vytvoření a zobrazení grafu
                fig = plot_predictions(predictions)
                st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Chyba při klasifikaci: {str(e)}")
            
            # Úklid dočasného souboru
            os.remove(temp_file)

if __name__ == "__main__":
    main()