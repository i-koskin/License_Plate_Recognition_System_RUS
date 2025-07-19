import math
from typing import List, Tuple, Dict, Any


def box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Вычисляет центр прямоугольника (bbox).

    :param box: Кортеж (x1, y1, x2, y2)
    :return: Кортеж координат центра (cx, cy)
    """
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def center_distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    """
    Вычисляет евклидово расстояние между двумя точками.

    :param c1: Координаты первой точки (x, y)
    :param c2: Координаты второй точки (x, y)
    :return: Расстояние между точками
    """
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def assign_plates_to_vehicles(
    plate_boxes: List[Tuple[float, float, float, float]],
    plate_texts: List[str],
    tracks: List[Tuple[Any, Any, Any, Any, int]]
) -> Dict[int, str]:
    """
    Связывает распознанные номера с ближайшими автомобилями по координатам.

    :param plate_boxes: Список bbox'ов номеров (x1, y1, x2, y2)
    :param plate_texts: Список распознанных текстов номеров
    :param tracks: Список треков с bbox и SID [(bbox, ..., ..., ..., sid)]
    :return: Словарь соответствий {SID: plate_text}
    """
    assignments: Dict[int, str] = {}
    used_sids = set()
    used_plates = set()

    for i, plate_box in enumerate(plate_boxes):
        plate_center = box_center(plate_box)
        best_dist = float("inf")
        best_sid = None
        best_vbox = None

        # Поиск ближайшего автомобиля
        for track in tracks:
            vehicle_box = track[0]
            sid = int(track[4])

            if sid in used_sids:
                continue

            vehicle_center = box_center(vehicle_box)
            dist = center_distance(plate_center, vehicle_center)
            diag = math.hypot(
                vehicle_box[2] - vehicle_box[0], vehicle_box[3] - vehicle_box[1])

            # Сравнение с допустимой дистанцией (в 5 раз меньше диагонали)
            if dist < best_dist and dist < 5.0 * diag:
                best_dist = dist
                best_sid = sid
                best_vbox = vehicle_box

        # Проверка, что центр номера лежит внутри bbox машины
        if best_sid is not None:
            vx1, vy1, vx2, vy2 = best_vbox
            in_box = (vx1 <= plate_center[0] <= vx2) and (
                vy1 <= plate_center[1] <= vy2)

            if in_box:
                plate_text = plate_texts[i].strip()
                if plate_text and plate_text not in used_plates:
                    assignments[best_sid] = plate_text
                    used_sids.add(best_sid)
                    used_plates.add(plate_text)

    return assignments
