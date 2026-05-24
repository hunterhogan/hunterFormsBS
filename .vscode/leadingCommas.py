from __future__ import annotations

import sys

def reformatTrailingCommasToLeadingCommas(text: str) -> str:
	lineEnding = '\r\n' if '\r\n' in text else '\n'
	lines = text.replace('\r\n', '\n').split('\n')

	for indexLine in range(len(lines) - 1):
		lineStrippedRight = lines[indexLine].rstrip()
		if not lineStrippedRight.endswith(','):
			continue
		lineFollowing = lines[indexLine + 1]
		lineFollowingStrippedLeft = lineFollowing.lstrip()
		if not lineFollowingStrippedLeft or lineFollowingStrippedLeft.startswith((']', ')', '}')):
			lines[indexLine] = lineStrippedRight[:-1]
		else:
			indentation = lineFollowing[:len(lineFollowing) - len(lineFollowingStrippedLeft)]
			lines[indexLine] = lineStrippedRight[:-1]
			lines[indexLine + 1] = indentation + ', ' + lineFollowingStrippedLeft

	lastLineStrippedRight = lines[-1].rstrip()
	if lastLineStrippedRight.endswith(','):
		lines[-1] = lastLineStrippedRight[:-1]

	return lineEnding.join(lines)


if __name__ == '__main__':
	sys.stdout.write(reformatTrailingCommasToLeadingCommas(sys.stdin.read()))
