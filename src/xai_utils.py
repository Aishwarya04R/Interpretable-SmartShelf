import numpy as np
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Try importing backends safely
try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

class ExplainableAI:
    """
    Universal HD XAI: Generates high-resolution Grad-CAM and LIME visualizations
    for TensorFlow (Meat/Fruit/Bakery) and PyTorch (Veg) models.
    Updated to ensure Bakery heatmap works and outputs match input size.
    """

    def __init__(self, model, classes, backend="tensorflow"):
        self.model = model
        self.classes = classes
        self.backend = backend # 'tensorflow' or 'pytorch'

    # ==========================
    # TENSORFLOW LOGIC
    # ==========================
    def _get_tf_last_conv(self):
        """Finds the best convolutional layer for high-detail gradients"""
        # 1. Specific Bakery Layer (CRITICAL CHECK)
        try:
            if self.model.get_layer('msff_last_conv'): return 'msff_last_conv'
        except: pass
            
        # 2. EfficientNet Naming (Meat)
        for layer in reversed(self.model.layers):
            if 'top_activation' in layer.name: return layer.name
            
        # 3. Fallback: Find last generic Conv2D
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D): return layer.name
            if 'conv' in layer.name.lower(): return layer.name
            
        return None

    def _grad_cam_tf(self, image_array):
        layer_name = self._get_tf_last_conv()
        if not layer_name: return None, None
        
        # Robustly handle input dimensions
        if image_array.ndim == 3: input_tensor = np.expand_dims(image_array, 0)
        else: input_tensor = image_array

        is_multi_head = isinstance(self.model.output, list) and len(self.model.output) > 1
        
        # Build Gradient Model
        if is_multi_head:
            grad_model = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[self.model.get_layer(layer_name).output, self.model.output[0], self.model.output[1]]
            )
        else:
            grad_model = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[self.model.get_layer(layer_name).output, self.model.output]
            )

        with tf.GradientTape(persistent=True) as tape:
            outputs = grad_model(input_tensor)
            conv_out = outputs[0]
            
            if is_multi_head:
                pred_id, pred_status = outputs[1], outputs[2]
                loss_id = pred_id[0][tf.argmax(pred_id[0])]
                loss_status = pred_status[0][tf.argmax(pred_status[0])]
            else:
                pred_status = outputs[1]
                # For Bakery (Regression), we use the output value itself
                # CRITICAL FIX: Ensure we get a scalar loss for gradients
                if pred_status.shape[-1] == 1: 
                    loss_status = pred_status[0][0]
                else: 
                    loss_status = pred_status[0][tf.argmax(pred_status[0])]
                loss_id = None

        # Generate Gradients
        grads_st = tape.gradient(loss_status, conv_out)
        
        # FIX FOR BAKERY: Use ABSOLUTE gradients for regression.
        # This highlights features that strongly affect the outcome (positively OR negatively).
        # This ensures Mold (negative impact) shows up as Red.
        if not is_multi_head and pred_status.shape[-1] == 1:
             pool_st = tf.reduce_mean(tf.abs(grads_st), axis=(0, 1, 2))
        else:
             pool_st = tf.reduce_mean(grads_st, axis=(0, 1, 2))
        
        # Weighted Combination (Status/Quality Heatmap)
        hm_st = tf.maximum(conv_out[0] @ pool_st[..., tf.newaxis], 0)
        
        # Robust Normalization
        hm_st = hm_st.numpy().squeeze()
        if np.max(hm_st) != 0:
            hm_st = (hm_st - np.min(hm_st)) / (np.max(hm_st) - np.min(hm_st) + 1e-10)
        
        # Identity Heatmap Logic
        hm_id = None
        if is_multi_head:
            grads_id = tape.gradient(loss_id, conv_out)
            pool_id = tf.reduce_mean(grads_id, axis=(0, 1, 2))
            hm_id = tf.maximum(conv_out[0] @ pool_id[..., tf.newaxis], 0)
            hm_id = hm_id.numpy().squeeze()
            if np.max(hm_id) != 0:
                hm_id = (hm_id - np.min(hm_id)) / (np.max(hm_id) - np.min(hm_id) + 1e-10)
        else:
            # For single head (Bakery), duplicate the valid heatmap so both slots work
            hm_id = hm_st

        del tape
        return hm_id, hm_st

    # ==========================
    # PYTORCH LOGIC (VEG)
    # ==========================
    def _grad_cam_pytorch(self, image_tensor):
        target_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d): target_layer = module
        if target_layer is None: return None, None

        gradients, activations = [], []
        def b_hook(m, i, o): gradients.append(o[0])
        def f_hook(m, i, o): activations.append(o)

        h1 = target_layer.register_forward_hook(f_hook)
        h2 = target_layer.register_full_backward_hook(b_hook)

        self.model.eval()
        preds = self.model(image_tensor)
        
        # Target Freshness Head (Index 1) for Multi-Task Veg
        if isinstance(preds, tuple): score = preds[1].max()
        else: score = preds.max()
            
        self.model.zero_grad()
        score.backward()
        h1.remove(); h2.remove()

                    
             
        grads = gradients[0].cpu().data.numpy()[0]
        fmap = activations[0].cpu().data.numpy()[0]
        weights = np.mean(grads, axis=(1, 2))

        heatmap = np.zeros(fmap.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * fmap[i, :, :]

        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-10)

        # Return single heatmap for PyTorch currently
        return None, heatmap


    # ==========================
    # PUBLIC METHODS
    # ==========================
    def grad_cam(self, image_input):
        if self.backend == "tensorflow":
            return self._grad_cam_tf(image_input)
        elif self.backend == "pytorch":
            # Convert HWC numpy to NCHW tensor for PyTorch
            tensor = torch.from_numpy(image_input).permute(2, 0, 1).unsqueeze(0).float()
            return self._grad_cam_pytorch(tensor)

    def visualize_grad_cam(self, image_array, heatmap, alpha=0.5):
        if heatmap is None: return image_array
        
        if image_array.ndim == 4: image_array = image_array[0]
            
        # Ensure image is 0-255 uint8
        if image_array.max() <= 1.0: img_u8 = (image_array * 255).astype(np.uint8)
        else: img_u8 = image_array.astype(np.uint8)
        
        # HD Resize (Cubic)
        heatmap = cv2.resize(heatmap, (img_u8.shape[1], img_u8.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend
        return cv2.addWeighted(img_u8, 1 - alpha, heatmap, alpha, 0)

    def lime_explanation(self, image_array, target_head=0, num_samples=30):
        explainer = lime_image.LimeImageExplainer()

        if image_array.ndim == 4: image_array = image_array[0]

        # TF Wrapper
        def predict_fn_tf(images):
            # Preprocess
            processed = tf.keras.applications.efficientnet.preprocess_input(images.astype(np.float32))
            preds = self.model.predict(processed, verbose=0)
            is_multi = isinstance(self.model.output, list) and len(self.model.output) > 1
            if is_multi: return preds[target_head]
            return preds

        # PyTorch Wrapper
        def predict_fn_torch(images):
            imgs_float = images.astype(np.float32) / 255.0
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
            imgs_norm = (imgs_float - mean) / std
            tensor = torch.from_numpy(imgs_norm).permute(0, 3, 1, 2).float()
            self.model.eval()
            with torch.no_grad():
                preds = self.model(tensor)
                if isinstance(preds, tuple): return preds[1].numpy()
                return preds.numpy()

        predict_fn = predict_fn_torch if self.backend == 'pytorch' else predict_fn_tf
        img_lime = image_array.astype(np.double)
        if img_lime.max() <= 1.0: img_lime *= 255.0

        # LIME for Regression/Single Class
        explanation = explainer.explain_instance(
            img_lime, predict_fn, top_labels=1, hide_color=0, num_samples=num_samples
        )
        return explanation

    def visualize_lime(self, explanation, label_idx, num_features=5):
        try:
            # Check if regression (no top_labels usually, or label is 0)
            # LIME for regression returns intercept/local_pred etc. 
            # explain_instance returns an Explanation object. 
            # For regression, we typically visualize the only available label (usually 0 if not specified, or we check top_labels)
            
            # Safe Access
            if explanation.mode == 'regression':
                 label = list(explanation.local_exp.keys())[0] # Get the only label key
            else:
                 label = label_idx if label_idx in explanation.local_exp else explanation.top_labels[0]

            temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=num_features, hide_rest=False)
        except Exception as e:
            # Fallback
            print(f"LIME Viz Error: {e}. Using default label.")
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=num_features, hide_rest=False)
            
        return mark_boundaries(temp / 255.0, mask, color=(1, 1, 0), mode='thick')
        
    def analyze_heatmap_focus(self, heatmap):
        if heatmap is None: return "General Area"
        h, w = heatmap.shape
        regions = {"Top": np.mean(heatmap[0:h//3, :]), "Center": np.mean(heatmap[h//3:2*h//3, :]), "Bottom": np.mean(heatmap[2*h//3:, :])}
        return max(regions, key=regions.get)
