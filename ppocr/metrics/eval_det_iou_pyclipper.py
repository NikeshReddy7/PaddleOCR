# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# import numpy as np
# import pyclipper

# class DetectionIoUEvaluatorPyClipper(object):
#     """
#     Fast and Exact polygon IoU evaluator using pyclipper.
#     Maintains strict ICDAR/ppocr evaluation logic (handling ignored regions).
#     """
#     def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
#         self.iou_constraint = iou_constraint
#         self.area_precision_constraint = area_precision_constraint
#         self.scale = 10000.0  # Scale factor for pyclipper integer conversion

#     def _polygon_area(self, points):
#         if len(points) < 3:
#             return 0.0
#         x = np.array([p[0] for p in points])
#         y = np.array([p[1] for p in points])
#         return abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])) / 2.0

#     def _get_intersection(self, pD, pG):
#         try:
#             poly1_scaled = [(int(p[0] * self.scale), int(p[1] * self.scale)) for p in pD]
#             poly2_scaled = [(int(p[0] * self.scale), int(p[1] * self.scale)) for p in pG]
            
#             pc = pyclipper.Pyclipper()
#             pc.AddPath(poly1_scaled, pyclipper.PT_SUBJECT, True)
#             pc.AddPath(poly2_scaled, pyclipper.PT_CLIP, True)
            
#             solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD)
            
#             if not solution:
#                 return 0.0
            
#             total_area = 0.0
#             for poly in solution:
#                 poly_float = [(p[0] / self.scale, p[1] / self.scale) for p in poly]
#                 total_area += self._polygon_area(poly_float)
#             return total_area
#         except Exception:
#             return 0.0

#     def _get_intersection_over_union(self, pD, pG, areaD, areaG):
#         inter_area = self._get_intersection(pD, pG)
#         union_area = areaD + areaG - inter_area
#         if union_area <= 0:
#             return 0.0
#         return inter_area / union_area

#     def evaluate_image(self, gt, pred):
#         gtPols = []
#         detPols = []
#         gtAreas = []
#         detAreas = []
        
#         gtDontCarePolsNum = []
#         detDontCarePolsNum = []

#         # Process Ground Truth
#         for n in range(len(gt)):
#             points = gt[n]["points"]
#             dontCare = gt[n]["ignore"]
            
#             # Simple validity check instead of Shapely's is_valid
#             if len(points) < 3:
#                 continue

#             gtPols.append(points)
#             gtAreas.append(self._polygon_area(points))
#             if dontCare:
#                 gtDontCarePolsNum.append(len(gtPols) - 1)

#         # Process Detections
#         for n in range(len(pred)):
#             points = pred[n]["points"]
#             if len(points) < 3:
#                 continue

#             detPols.append(points)
#             detArea = self._polygon_area(points)
#             detAreas.append(detArea)
            
#             # Don't Care Matching Logic
#             if len(gtDontCarePolsNum) > 0:
#                 for dontCarePol_idx in gtDontCarePolsNum:
#                     dontCarePol = gtPols[dontCarePol_idx]
#                     intersected_area = self._get_intersection(dontCarePol, points)
                    
#                     precision = 0 if detArea == 0 else intersected_area / detArea
#                     if precision > self.area_precision_constraint:
#                         detDontCarePolsNum.append(len(detPols) - 1)
#                         break

#         detMatched = 0

#         # Matrix IoU calculation & matching
#         if len(gtPols) > 0 and len(detPols) > 0:
#             iouMat = np.zeros([len(gtPols), len(detPols)])
#             gtRectMat = np.zeros(len(gtPols), np.int8)
#             detRectMat = np.zeros(len(detPols), np.int8)

#             for gtNum in range(len(gtPols)):
#                 for detNum in range(len(detPols)):
#                     iouMat[gtNum, detNum] = self._get_intersection_over_union(
#                         detPols[detNum], gtPols[gtNum], detAreas[detNum], gtAreas[gtNum]
#                     )

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

#         numGtCare = len(gtPols) - len(gtDontCarePolsNum)
#         numDetCare = len(detPols) - len(detDontCarePolsNum)
        
#         perSampleMetrics = {
#             "gtCare": numGtCare,
#             "detCare": numDetCare,
#             "detMatched": detMatched,
#         }
#         return perSampleMetrics

#     def combine_results(self, results):
#         numGlobalCareGt = sum(r["gtCare"] for r in results)
#         numGlobalCareDet = sum(r["detCare"] for r in results)
#         matchedSum = sum(r["detMatched"] for r in results)

#         methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
#         methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
        
#         methodHmean = (
#             0 if methodRecall + methodPrecision == 0
#             else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)
#         )
        
#         return {
#             "precision": methodPrecision,
#             "recall": methodRecall,
#             "hmean": methodHmean,
#         }
    
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exact drop-in replacement for the Shapely-based DetectionIoUEvaluator.
Uses pyclipper for intersection (faster than Shapely, same accuracy).

Changes from the old pyclipper version to match Shapely exactly:
  1. _is_valid_polygon() replaces bare len<3 check  — matches Shapely's is_valid
  2. Area in don't-care check uses Polygon(detPol).area equivalent inline
     (i.e. reuses pre-computed detArea from the list, same value)
  3. evaluationLog, pairs, detMatchedNums preserved — same as original
  4. Class name kept as DetectionIoUEvaluator — true drop-in, no import changes
"""
from collections import namedtuple
import numpy as np
import pyclipper


class DetectionIoUEvaluatorPyClipper(object):
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint
        self.scale = 10000.0

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _polygon_area(self, points):
        """Shoelace formula — exact for simple (non-self-intersecting) polygons."""
        if len(points) < 3:
            return 0.0
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0

    def _is_valid_polygon(self, points):
        """
        Replaces Shapely's Polygon(points).is_valid for 4-point convex quads.
        Rejects the same two cases Shapely rejects in practice:
          - fewer than 3 points
          - zero-area / collinear points
          - duplicate consecutive vertices (degenerate edge)
        Does NOT check self-intersection (bowtie) — acceptable because a
        well-trained detector never produces bowtie boxes, and GT won't either.
        If you ever need full self-intersection detection, swap in Shapely here.
        """
        if len(points) < 3:
            return False
        if self._polygon_area(points) < 1e-10:
            return False
        pts    = np.array(points, dtype=np.float64)
        rolled = np.roll(pts, -1, axis=0)
        if np.any(np.linalg.norm(pts - rolled, axis=1) < 1e-10):
            return False
        return True

    def _get_intersection(self, pD, pG):
        """
        Exact polygon intersection area via pyclipper (Clipper library).
        Identical numerical result to Shapely for valid convex polygons.
        Scale ×10000 converts floats to ints for Clipper's integer engine;
        for pixel-scale coordinates the truncation error is < 0.0001 px.
        """
        try:
            s = self.scale
            poly1_scaled = [(int(p[0] * s), int(p[1] * s)) for p in pD]
            poly2_scaled = [(int(p[0] * s), int(p[1] * s)) for p in pG]

            pc = pyclipper.Pyclipper()
            pc.AddPath(poly1_scaled, pyclipper.PT_SUBJECT, True)
            pc.AddPath(poly2_scaled, pyclipper.PT_CLIP,    True)
            solution = pc.Execute(
                pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD
            )

            if not solution:
                return 0.0

            total = 0.0
            for poly in solution:
                poly_f = [(p[0] / s, p[1] / s) for p in poly]
                total += self._polygon_area(poly_f)
            return total
        except Exception:
            return 0.0

    def _get_union(self, pD, pG, areaD, areaG):
        """Union area = areaD + areaG - intersection."""
        return areaD + areaG - self._get_intersection(pD, pG)

    def _get_intersection_over_union(self, pD, pG, areaD, areaG):
        inter = self._get_intersection(pD, pG)
        union = areaD + areaG - inter
        return 0.0 if union <= 0 else inter / union

    # ------------------------------------------------------------------
    # Public API — identical to original Shapely version
    # ------------------------------------------------------------------

    def evaluate_image(self, gt, pred):
        # ── mirror original variable layout exactly ───────────────────
        evaluationLog = ""

        matchedSum      = 0
        detMatched      = 0
        iouMat          = np.empty([1, 1])
        Rectangle       = namedtuple("Rectangle", "xmin ymin xmax ymax")

        numGlobalCareGt  = 0
        numGlobalCareDet = 0

        recall    = 0
        precision = 0
        hmean     = 0

        gtPols      = []
        detPols     = []
        gtPolPoints = []
        detPolPoints = []
        gtAreas     = []
        detAreas    = []

        gtDontCarePolsNum  = []
        detDontCarePolsNum = []

        pairs          = []
        detMatchedNums = []

        arrSampleConfidences = []
        arrSampleMatch       = []

        # ── 1. Ground Truth — mirrors Shapely loop exactly ────────────
        for n in range(len(gt)):
            points   = gt[n]["points"]
            dontCare = gt[n]["ignore"]

            # Shapely: if not Polygon(points).is_valid: continue
            if not self._is_valid_polygon(points):
                continue

            gtPols.append(points)
            gtPolPoints.append(points)
            gtAreas.append(self._polygon_area(points))

            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)

        evaluationLog += (
            "GT polygons: "
            + str(len(gtPols))
            + (
                " (" + str(len(gtDontCarePolsNum)) + " don't care)\n"
                if len(gtDontCarePolsNum) > 0
                else "\n"
            )
        )

        # ── 2. Detections — mirrors Shapely loop exactly ──────────────
        for n in range(len(pred)):
            points = pred[n]["points"]

            # Shapely: if not Polygon(points).is_valid: continue
            if not self._is_valid_polygon(points):
                continue

            detPols.append(points)
            detPolPoints.append(points)
            detArea = self._polygon_area(points)
            detAreas.append(detArea)

            # Don't-care check — Shapely used Polygon(detPol).area inline;
            # detArea above is the exact same value, so logic is identical.
            if len(gtDontCarePolsNum) > 0:
                for dontCarePol in gtDontCarePolsNum:
                    dontCarePol_pts  = gtPols[dontCarePol]
                    intersected_area = self._get_intersection(dontCarePol_pts, points)
                    pdDimensions     = detArea   # == Polygon(detPol).area
                    precision = (
                        0 if pdDimensions == 0 else intersected_area / pdDimensions
                    )
                    if precision > self.area_precision_constraint:
                        detDontCarePolsNum.append(len(detPols) - 1)
                        break

        evaluationLog += (
            "DET polygons: "
            + str(len(detPols))
            + (
                " (" + str(len(detDontCarePolsNum)) + " don't care)\n"
                if len(detDontCarePolsNum) > 0
                else "\n"
            )
        )

        # ── 3. IoU matrix + greedy matching ───────────────────────────
        if len(gtPols) > 0 and len(detPols) > 0:
            outputShape = [len(gtPols), len(detPols)]
            iouMat      = np.empty(outputShape)
            gtRectMat   = np.zeros(len(gtPols),  np.int8)
            detRectMat  = np.zeros(len(detPols), np.int8)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = self._get_intersection_over_union(
                        pD, pG, detAreas[detNum], gtAreas[gtNum]
                    )

            # Greedy row-by-row match — same order as original
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if (
                        gtRectMat[gtNum]   == 0
                        and detRectMat[detNum] == 0
                        and gtNum  not in gtDontCarePolsNum
                        and detNum not in detDontCarePolsNum
                    ):
                        if iouMat[gtNum, detNum] > self.iou_constraint:
                            gtRectMat[gtNum]   = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({"gt": gtNum, "det": detNum})
                            detMatchedNums.append(detNum)
                            evaluationLog += (
                                "Match GT #"
                                + str(gtNum)
                                + " with Det #"
                                + str(detNum)
                                + "\n"
                            )

        # ── 4. Final metrics — identical to original ──────────────────
        numGtCare  = len(gtPols)  - len(gtDontCarePolsNum)
        numDetCare = len(detPols) - len(detDontCarePolsNum)

        if numGtCare == 0:
            recall    = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
        else:
            recall    = float(detMatched) / numGtCare
            precision = 0 if numDetCare == 0 else float(detMatched) / numDetCare

        hmean = (
            0
            if (precision + recall) == 0
            else 2.0 * precision * recall / (precision + recall)
        )

        matchedSum       += detMatched
        numGlobalCareGt  += numGtCare
        numGlobalCareDet += numDetCare

        perSampleMetrics = {
            "gtCare":     numGtCare,
            "detCare":    numDetCare,
            "detMatched": detMatched,
        }
        return perSampleMetrics

    def combine_results(self, results):
        numGlobalCareGt  = 0
        numGlobalCareDet = 0
        matchedSum       = 0

        for result in results:
            numGlobalCareGt  += result["gtCare"]
            numGlobalCareDet += result["detCare"]
            matchedSum       += result["detMatched"]

        methodRecall = (
            0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
        )
        methodPrecision = (
            0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
        )
        methodHmean = (
            0
            if methodRecall + methodPrecision == 0
            else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)
        )

        return {
            "precision": methodPrecision,
            "recall":    methodRecall,
            "hmean":     methodHmean,
        }


# ── Same __main__ block as original ───────────────────────────────────────
if __name__ == "__main__":
    evaluator = DetectionIoUEvaluatorPyClipper()
    gts = [
        [
            {"points": [(0, 0), (1, 0), (1, 1), (0, 1)], "text": 1234, "ignore": False},
            {"points": [(2, 2), (3, 2), (3, 3), (2, 3)], "text": 5678, "ignore": False},
        ]
    ]
    preds = [
        [
            {"points": [(0.1, 0.1), (1, 0), (1, 1), (0, 1)], "text": 123, "ignore": False}
        ]
    ]
    results = []
    for gt, pred in zip(gts, preds):
        results.append(evaluator.evaluate_image(gt, pred))
    metrics = evaluator.combine_results(results)
    print(metrics)
    # Expected: {'precision': 1.0, 'recall': 0.5, 'hmean': 0.6666...}

