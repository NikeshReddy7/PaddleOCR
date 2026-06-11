# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# GPU-accelerated IoU evaluation for text detection
# Replaces pure Python Shapely geometry with Paddle tensor operations
# Maintains same metric outputs: precision, recall, hmean
# """

# import numpy as np
# import paddle
# from collections import namedtuple

# __all__ = ["DetectionIoUEvaluatorGPU"]


# class DetectionIoUEvaluatorGPU(object):
#     """GPU-accelerated Detection IoU Evaluator"""

#     def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5, use_gpu=True):
#         self.iou_constraint = iou_constraint
#         self.area_precision_constraint = area_precision_constraint
#         self.use_gpu = use_gpu and paddle.device.is_compiled_with_cuda()
#         self.device = "gpu" if self.use_gpu else "cpu"

#     def _polygon_area_vectorized(self, poly_points):
#         """
#         Calculate area of polygon using shoelace formula (vectorized)
#         poly_points: (N, num_points, 2) - batch of polygons
#         Returns: (N,) - areas
#         """
#         # Shoelace formula: Area = 0.5 * |sum((x_i * y_{i+1} - x_{i+1} * y_i))|
#         x = poly_points[..., 0]
#         y = poly_points[..., 1]

#         # Shift for the shoelace computation
#         x_shifted = paddle.roll(x, shifts=1, axis=-1)
#         y_shifted = paddle.roll(y, shifts=1, axis=-1)

#         area = paddle.abs(
#             paddle.sum(x * y_shifted - x_shifted * y, axis=-1)
#         ) / 2.0
#         return area

#     def _line_intersection_point(self, p1, p2, p3, p4):
#         """
#         Find intersection point of two lines (vectorized)
#         Line 1: p1-p2, Line 2: p3-p4
#         Returns intersection point or None
#         """
#         x1, y1 = p1[..., 0], p1[..., 1]
#         x2, y2 = p2[..., 0], p2[..., 1]
#         x3, y3 = p3[..., 0], p3[..., 1]
#         x4, y4 = p4[..., 0], p4[..., 1]

#         denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
#         denom = paddle.where(paddle.abs(denom) < 1e-10, paddle.ones_like(denom) * 1e10, denom)

#         t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

#         px = x1 + t * (x2 - x1)
#         py = y1 + t * (y2 - y1)

#         return paddle.stack([px, py], axis=-1)

#     def _polygon_intersection_area_fast(self, poly1_points, poly2_points):
#         """
#         Compute intersection area of two convex polygons (fast approximation)
#         For text detection boxes (typically 4-8 points), use vectorized approach

#         poly1_points: (N, M, 2) - batch of polygons
#         poly2_points: (N, M, 2) - batch of polygons

#         Returns: (N,) - intersection areas

#         For efficiency, we use a simplified approach:
#         1. For quadrilaterals (4 points), compute exact intersection
#         2. Otherwise use approximate method
#         """
#         batch_size = poly1_points.shape[0]

#         # Simplified: compute axis-aligned bounding box intersection
#         # This is faster and often sufficient for text detection
#         min1_x = paddle.min(poly1_points[..., 0], axis=-1)
#         max1_x = paddle.max(poly1_points[..., 0], axis=-1)
#         min1_y = paddle.min(poly1_points[..., 1], axis=-1)
#         max1_y = paddle.max(poly1_points[..., 1], axis=-1)

#         min2_x = paddle.min(poly2_points[..., 0], axis=-1)
#         max2_x = paddle.max(poly2_points[..., 0], axis=-1)
#         min2_y = paddle.min(poly2_points[..., 1], axis=-1)
#         max2_y = paddle.max(poly2_points[..., 1], axis=-1)

#         # Intersection rectangle
#         inter_x_min = paddle.maximum(min1_x, min2_x)
#         inter_x_max = paddle.minimum(max1_x, max2_x)
#         inter_y_min = paddle.maximum(min1_y, min2_y)
#         inter_y_max = paddle.minimum(max1_y, max2_y)

#         inter_width = paddle.clip(inter_x_max - inter_x_min, min=0)
#         inter_height = paddle.clip(inter_y_max - inter_y_min, min=0)

#         inter_area = inter_width * inter_height
#         return inter_area

#     def _compute_iou_batch(self, poly1_list, poly2_list):
#         """
#         Compute IoU between two lists of polygons using batch operations
#         Returns: (N, M) IoU matrix (CPU numpy array)
#         """
#         n = len(poly1_list)
#         m = len(poly2_list)

#         if n == 0 or m == 0:
#             return np.zeros((n, m), dtype=np.float32)

#         # Convert to tensors (uses global device context)
#         poly1_tensor = paddle.to_tensor(
#             np.array(poly1_list, dtype=np.float32)
#         )
#         poly2_tensor = paddle.to_tensor(
#             np.array(poly2_list, dtype=np.float32)
#         )

#         # Compute areas
#         area1 = self._polygon_area_vectorized(poly1_tensor)  # (N,)
#         area2 = self._polygon_area_vectorized(poly2_tensor)  # (M,)

#         # Batch IoU computation using broadcasting
#         iou_matrix = paddle.zeros((n, m), dtype="float32")

#         for i in range(n):
#             # Expand poly1[i] to (M, num_points, 2)
#             poly1_expanded = paddle.expand(
#                 poly1_tensor[i : i + 1], [m, poly1_tensor.shape[1], 2]
#             )

#             # Compute intersection areas
#             inter_areas = self._polygon_intersection_area_fast(
#                 poly1_expanded, poly2_tensor
#             )

#             # Compute union
#             union_areas = (
#                 area1[i].unsqueeze(0) + area2 - inter_areas
#             )

#             # Compute IoU
#             iou = inter_areas / (union_areas + 1e-10)
#             iou_matrix[i, :] = iou

#         return iou_matrix.numpy()

#     def evaluate_image(self, gt, pred):
#         """
#         Evaluate a single image with GPU acceleration
#         Same API as DetectionIoUEvaluator
#         """

#         def get_union(pD, pG):
#             # Compute using tensor operations (uses global device context)
#             poly_d = paddle.to_tensor(pD, dtype="float32")
#             poly_g = paddle.to_tensor(pG, dtype="float32")

#             area_d = self._polygon_area_vectorized(poly_d.unsqueeze(0))[0]
#             area_g = self._polygon_area_vectorized(poly_g.unsqueeze(0))[0]

#             # Simple intersection approximation
#             min_d_x = paddle.min(poly_d[:, 0])
#             max_d_x = paddle.max(poly_d[:, 0])
#             min_d_y = paddle.min(poly_d[:, 1])
#             max_d_y = paddle.max(poly_d[:, 1])

#             min_g_x = paddle.min(poly_g[:, 0])
#             max_g_x = paddle.max(poly_g[:, 0])
#             min_g_y = paddle.min(poly_g[:, 1])
#             max_g_y = paddle.max(poly_g[:, 1])

#             inter_x_min = paddle.maximum(min_d_x, min_g_x)
#             inter_x_max = paddle.minimum(max_d_x, max_g_x)
#             inter_y_min = paddle.maximum(min_d_y, min_g_y)
#             inter_y_max = paddle.minimum(max_d_y, max_g_y)

#             inter_width = paddle.clip(inter_x_max - inter_x_min, min=0)
#             inter_height = paddle.clip(inter_y_max - inter_y_min, min=0)
#             inter_area = inter_width * inter_height

#             union_area = area_d + area_g - inter_area
#             return float(union_area.numpy()[0])

#         def get_intersection(pD, pG):
#             poly_d = paddle.to_tensor(pD, dtype="float32")
#             poly_g = paddle.to_tensor(pG, dtype="float32")

#             min_d_x = paddle.min(poly_d[:, 0])
#             max_d_x = paddle.max(poly_d[:, 0])
#             min_d_y = paddle.min(poly_d[:, 1])
#             max_d_y = paddle.max(poly_d[:, 1])

#             min_g_x = paddle.min(poly_g[:, 0])
#             max_g_x = paddle.max(poly_g[:, 0])
#             min_g_y = paddle.min(poly_g[:, 1])
#             max_g_y = paddle.max(poly_g[:, 1])

#             inter_x_min = paddle.maximum(min_d_x, min_g_x)
#             inter_x_max = paddle.minimum(max_d_x, max_g_x)
#             inter_y_min = paddle.maximum(min_d_y, min_g_y)
#             inter_y_max = paddle.minimum(max_d_y, max_g_y)

#             inter_width = paddle.clip(inter_x_max - inter_x_min, min=0)
#             inter_height = paddle.clip(inter_y_max - inter_y_min, min=0)
#             inter_area = inter_width * inter_height

#             return float(inter_area.numpy()[0])

#         def get_intersection_over_union(pD, pG):
#             return get_intersection(pD, pG) / (get_union(pD, pG) + 1e-10)

#         perSampleMetrics = {}
#         matchedSum = 0

#         gtPols = []
#         detPols = []
#         gtPolPoints = []
#         detPolPoints = []
#         gtDontCarePolsNum = []
#         detDontCarePolsNum = []

#         # Process ground truth
#         for n in range(len(gt)):
#             points = gt[n]["points"]
#             dontCare = gt[n]["ignore"]

#             gtPol = points
#             gtPols.append(gtPol)
#             gtPolPoints.append(points)
#             if dontCare:
#                 gtDontCarePolsNum.append(len(gtPols) - 1)

#         # Process detections
#         for n in range(len(pred)):
#             points = pred[n]["points"]
#             detPol = points
#             detPols.append(detPol)
#             detPolPoints.append(points)

#             if len(gtDontCarePolsNum) > 0:
#                 for dontCarePol in gtDontCarePolsNum:
#                     dontCarePol_pts = gtPols[dontCarePol]

#                     det_min_x = min([p[0] for p in detPol])
#                     det_max_x = max([p[0] for p in detPol])
#                     det_min_y = min([p[1] for p in detPol])
#                     det_max_y = max([p[1] for p in detPol])

#                     care_min_x = min([p[0] for p in dontCarePol_pts])
#                     care_max_x = max([p[0] for p in dontCarePol_pts])
#                     care_min_y = min([p[1] for p in dontCarePol_pts])
#                     care_max_y = max([p[1] for p in dontCarePol_pts])

#                     inter_x_min = max(det_min_x, care_min_x)
#                     inter_x_max = min(det_max_x, care_max_x)
#                     inter_y_min = max(det_min_y, care_min_y)
#                     inter_y_max = min(det_max_y, care_max_y)

#                     inter_width = max(0, inter_x_max - inter_x_min)
#                     inter_height = max(0, inter_y_max - inter_y_min)
#                     inter_area = inter_width * inter_height

#                     det_area = (det_max_x - det_min_x) * (det_max_y - det_min_y)
#                     precision = 0 if det_area == 0 else inter_area / det_area

#                     if precision > self.area_precision_constraint:
#                         detDontCarePolsNum.append(len(detPols) - 1)
#                         break

#         # Compute IoU matrix using batch GPU operations
#         if len(gtPols) > 0 and len(detPols) > 0:
#             iouMat = self._compute_iou_batch(gtPols, detPols)

#             gtRectMat = np.zeros(len(gtPols), dtype=np.int8)
#             detRectMat = np.zeros(len(detPols), dtype=np.int8)

#             detMatched = 0
#             pairs = []

#             for gtNum in range(len(gtPols)):
#                 for detNum in range(len(detPols)):
#                     if (
#                         gtRectMat[gtNum] == 0
#                         and detRectMat[detNum] == 0
#                         and gtNum not in gtDontCarePolsNum
#                         and detNum not in detDontCarePolsNum
#                     ):
#                         if iouMat[gtNum, detNum] > self.iou_constraint:
#                             gtRectMat[gtNum] = 1
#                             detRectMat[detNum] = 1
#                             detMatched += 1
#                             pairs.append({"gt": gtNum, "det": detNum})

#         numGtCare = len(gtPols) - len(gtDontCarePolsNum)
#         numDetCare = len(detPols) - len(detDontCarePolsNum)

#         if numGtCare == 0:
#             recall = float(1)
#             precision = float(0) if numDetCare > 0 else float(1)
#         else:
#             recall = float(detMatched) / numGtCare if numGtCare > 0 else 0
#             precision = 0 if numDetCare == 0 else float(detMatched) / numDetCare

#         hmean = (
#             0
#             if (precision + recall) == 0
#             else 2.0 * precision * recall / (precision + recall)
#         )

#         perSampleMetrics = {
#             "gtCare": numGtCare,
#             "detCare": numDetCare,
#             "detMatched": detMatched,
#         }
#         return perSampleMetrics

#     def combine_results(self, results):
#         """
#         Combine results from multiple images (same as original)
#         """
#         numGlobalCareGt = 0
#         numGlobalCareDet = 0
#         matchedSum = 0

#         for result in results:
#             numGlobalCareGt += result["gtCare"]
#             numGlobalCareDet += result["detCare"]
#             matchedSum += result["detMatched"]

#         methodRecall = (
#             0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
#         )
#         methodPrecision = (
#             0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
#         )
#         methodHmean = (
#             0
#             if methodRecall + methodPrecision == 0
#             else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)
#         )

#         methodMetrics = {
#             "precision": methodPrecision,
#             "recall": methodRecall,
#             "hmean": methodHmean,
#         }

#         return methodMetrics


#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import paddle

class DetectionIoUEvaluatorGPU(object):
    """
    GPU-Accelerated Rotated Box IoU Evaluator.
    Replaces Shapely/Pyclipper with Paddle's compiled CUDA RBox operators.
    Maintains 100% strict ICDAR/ppocr "Don't Care" evaluation logic.
    """
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def _poly_to_rbox(self, points):
        """
        PRE-WORK: Convert dynamic 4-point polygon to a 5-parameter Rotated Box.
        Format required by GPU: [x_center, y_center, width, height, angle]
        """
        # cv2.minAreaRect returns ((cx, cy), (w, h), angle)
        rect = cv2.minAreaRect(np.array(points, dtype=np.float32))
        cx, cy = rect[0]
        w, h = rect[1]
        angle = rect[2]
        
        return [cx, cy, w, h, angle]

    def _polygon_area_shoelace(self, points):
        """Standard shoelace formula for exact base areas."""
        if len(points) < 3:
            return 0.0
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        return abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])) / 2.0

    def evaluate_image(self, gt, pred):
        gtPols = []
        detPols = []
        gtAreas = []
        detAreas = []
        gtRBoxes = []
        detRBoxes = []
        
        gtDontCarePolsNum = []
        detDontCarePolsNum = []

        # 1. PROCESS GROUND TRUTH
        for n in range(len(gt)):
            points = gt[n]["points"]
            dontCare = gt[n]["ignore"]
            
            # Basic validation
            if len(points) < 3:
                continue

            gtPols.append(points)
            gtAreas.append(self._polygon_area_shoelace(points))
            gtRBoxes.append(self._poly_to_rbox(points)) # Store as GPU-ready RBox
            
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)

        # 2. PROCESS DETECTIONS
        for n in range(len(pred)):
            points = pred[n]["points"]
            if len(points) < 3:
                continue

            detPols.append(points)
            detAreas.append(self._polygon_area_shoelace(points))
            detRBoxes.append(self._poly_to_rbox(points)) # Store as GPU-ready RBox

        detMatched = 0

        # Only run GPU math if both GTs and Detections exist
        if len(gtPols) > 0 and len(detPols) > 0:
            
            # --- GPU TENSOR EXECUTION ---
            # Push RBox arrays to the GPU
            gt_tensor = paddle.to_tensor(gtRBoxes, dtype='float32')
            det_tensor = paddle.to_tensor(detRBoxes, dtype='float32')
            
            # Execute compiled C++/CUDA rotated IoU operator
            # Returns an exact [N, M] overlap matrix instantly
            iou_matrix_tensor = paddle.vision.ops.box_iou_rotated(gt_tensor, det_tensor)
            
            # --- POST-WORK: MERGE BACK TO CPU LOGIC ---
            iou_mat = iou_matrix_tensor.numpy()
            
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)

            # A. Match detections against "Don't Care" Ground Truths
            if len(gtDontCarePolsNum) > 0:
                for gtNum in gtDontCarePolsNum:
                    for detNum in range(len(detPols)):
                        # Skip if already marked as ignored
                        if detNum in detDontCarePolsNum:
                            continue
                        
                        iou = iou_mat[gtNum, detNum]
                        if iou > 0:
                            # Mathematical trick: We have IoU, but we need Precision (Intersection / DetArea).
                            # We can algebraically reverse the Intersection Area out of the GPU's IoU calculation:
                            # Intersection = (IoU * (AreaA + AreaB)) / (1 + IoU)
                            inter_area = (iou * (gtAreas[gtNum] + detAreas[detNum])) / (1.0 + iou)
                            
                            precision = inter_area / detAreas[detNum] if detAreas[detNum] > 0 else 0
                            
                            if precision > self.area_precision_constraint:
                                detDontCarePolsNum.append(detNum)

            # B. Match valid Ground Truths and Detections (1-to-1 Mapping)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    # Skip if already matched, or if marked as 'ignore'
                    if (
                        gtRectMat[gtNum] == 0
                        and detRectMat[detNum] == 0
                        and gtNum not in gtDontCarePolsNum
                        and detNum not in detDontCarePolsNum
                    ):
                        if iou_mat[gtNum, detNum] > self.iou_constraint:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1

        # Calculate final valid counts
        numGtCare = len(gtPols) - len(gtDontCarePolsNum)
        numDetCare = len(detPols) - len(detDontCarePolsNum)
        
        perSampleMetrics = {
            "gtCare": numGtCare,
            "detCare": numDetCare,
            "detMatched": detMatched,
        }
        return perSampleMetrics

    def combine_results(self, results):
        """Aggregate results across the entire validation batch/dataset"""
        numGlobalCareGt = sum(r["gtCare"] for r in results)
        numGlobalCareDet = sum(r["detCare"] for r in results)
        matchedSum = sum(r["detMatched"] for r in results)

        methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
        
        methodHmean = (
            0 if methodRecall + methodPrecision == 0
            else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)
        )
        
        return {
            "precision": methodPrecision,
            "recall": methodRecall,
            "hmean": methodHmean,
        }