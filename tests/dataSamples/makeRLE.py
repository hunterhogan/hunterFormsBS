from __future__ import annotations

from hunterMakesPy.dataStructures import autoDecodingRLE
from sys import stdout
from typing import Any, TYPE_CHECKING
import numpy

if TYPE_CHECKING:
    from numpy.typing import NDArray

aList: list[int] = [28] + [24] * 14 + [28] * 3 + [36] * 3 + [40, 40, 44] + [52] * 3 + [60, 64, 68, 76, 80, 80, 88, 96, 104, 112, 116, 124, 132, 144, 156, 164, 176, 188, 200, 216, 228, 244, 264, 284, 304, 320, 344, 372, 396, 420, 452, 488, 520]

anArray: NDArray[Any] = numpy.array(aList)

aString: str = autoDecodingRLE(anArray, assumeAddSpaces=True)

stdout.write(aString)
