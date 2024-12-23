from typing import Optional


def convert_bbox_format(coords, in_format: str = 'xywh', out_format: str = 'xyxy', width: Optional[float] = None, height: Optional[float] = None):
    if in_format == 'whole':
        assert width is not None
        assert height is not None

        coords = [0, width, 0, height]
    
    if in_format == 'xywh' and out_format == 'xyxy':
        coords[2] += coords[0]
        coords[3] += coords[1]

    if in_format == 'xyxy' and out_format == 'xywh':
        coords[2] -= coords[0]
        coords[3] -= coords[1]

    if in_format == 'xyxy_01' and out_format == 'xyxy':
        assert width is not None
        assert height is not None

        coords[0] *= width
        coords[1] *= height
        coords[2] *= width
        coords[3] *= height

    return coords
