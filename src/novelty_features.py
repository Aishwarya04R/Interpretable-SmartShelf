import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd

class ConfidenceExplainer:
    @staticmethod
    def explain_confidence(confidence, food_type):
        if confidence > 0.95:
            return f"🎯 **Extremely Confident**: The AI is {confidence*100:.1f}% certain this is {food_type}.", "Very High", "#06d6a0"
        elif confidence > 0.85:
            return f"✅ **Highly Confident**: The AI is {confidence*100:.1f}% sure this is {food_type}.", "High", "#26de81"
        elif confidence > 0.70:
            return f"⚠️ **Moderately Confident**: The AI is {confidence*100:.1f}% confident.", "Moderate", "#fed330"
        else:
            return f"❓ **Low Confidence**: The AI is unsure. Check lighting.", "Low", "#ef476f"

class QualityDecayPredictor:
    @staticmethod
    def predict_decay_curve(current_score, shelf_life_days):
        # Handle 0 days case to prevent math errors
        if shelf_life_days < 0.1: shelf_life_days = 0.5

        hours = np.linspace(0, shelf_life_days * 24, 100)
        days = hours / 24
        decay_rate = -np.log(0.1) / (shelf_life_days * 24)
        quality_scores = current_score * 100 * np.exp(-decay_rate * hours)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=quality_scores, mode='lines', name='Quality', line=dict(color='#667eea', width=3), fill='tozeroy'))
        fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Safety Threshold")
        fig.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="Spoiled")

        fig.update_layout(title="<b>Predicted Quality Decay</b>", xaxis_title="Days", yaxis_title="Score", yaxis_range=[0, 105], height=350, margin=dict(l=20, r=20, t=40, b=20))
        return fig

class SmartStorageAdvisor:
    @staticmethod
    def generate_recommendations(food_type, score, days):
        recs = []
        if score < 40:
            recs.append(("❌", "Do Not Consume", "Food appears spoiled."))
        elif days < 1.0:
            recs.append(("🍳", "Cook Immediately", "Shelf life is critical."))
        else:
            recs.append(("❄️", "Refrigerate", f"Keep {food_type} at 0-4°C."))
            if days > 2:
                recs.append(("🧊", "Freeze", "Freeze to extend life by months."))
        return recs

class CarbonFootprintCalculator:
    CO2_MAP = {'fish': 5.5, 'chicken': 6.9, 'beef': 27.0, 'pork': 12.1, 'mutton': 39.2, 'prawn': 18.0, 'crab': 15.5}

    @staticmethod
    def calculate_impact(food_type, weight_kg):
        # Default to a safe fallback key if specific key missing
        base_type = food_type.lower().split('_')[0] # handle 'Apple_Fresh'
        factor = CarbonFootprintCalculator.CO2_MAP.get(base_type, 0.5) # Default 0.5 for Fruit/Veg

        co2 = factor * weight_kg
        miles = co2 / 0.404
        return co2, miles

    @staticmethod
    def generate_impact_text(food_type, co2_kg):
        """Generates contextual explanation for carbon footprint"""
        phones = int(co2_kg / 0.015) # Approx co2 to charge phone

        severity = ""
        if food_type.lower() in ['beef', 'mutton']:
            severity = "This is a **High-Impact** protein source."
        elif food_type.lower() in ['pork', 'prawn', 'crab', 'chicken', 'fish']:
            severity = "This is a **Medium-Impact** food source."
        else:
            severity = "While **Lower-Impact**, wasting this still affects the environment."

        return f"""
        **Environmental Context:**
        Wasting this amount of **{food_type}** generates **{co2_kg:.2f} kg of CO₂e**.
        To put this in perspective, that is equivalent to the energy required to charge a smartphone **{phones} times**.
        <br><br>
        {severity} **SmartShelf AI** helps you consume this before expiry to prevent this unnecessary climate impact.
        """

class XAITextGenerator:
    """Generates text explanations for AI Decisions"""

    @staticmethod
    def explain_gradcam(food_name, status, focus_region):
        """Generates text for Heatmaps"""
        if 'spoil' in status.lower() or 'rot' in status.lower():
            reason = "discoloration, surface texture changes, or microbial growth patterns"
        else:
            reason = "typical freshness markers like consistent color and firm texture"

        return f"""
        **Visual Attention Analysis:**
        The AI focused heavily on the **{focus_region}** of the image.
        For **{food_name}**, high activation in this area suggests the model detected {reason}
        that strongly correlate with the **{status}** classification.
        """

    @staticmethod
    def explain_lime(food_name, status):
        """Generates text for LIME Boundaries"""
        return f"""
        **Feature Boundary Analysis:**
        The yellow outlines mark the 'Super-Pixels' that positively influenced the decision.
        The AI isolated these specific segments as the most critical visual evidence for classifying this
        **{food_name}** as **{status}**. If these regions contain spots or slime, the prediction is highly reliable.
        """
