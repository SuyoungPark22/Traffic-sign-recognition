import math

class DimOrder(object):
	"""
	Represents the order of the dimensions in a dataset's shape.
	"""
	ChannelHeightWidth = ['c', 'h', 'w']
	HeightWidthChannel = ['h', 'w', 'c']


class SlidingWindow(object):
	"""
	Represents a single window into a larger dataset.
	"""
	
	def __init__(self, x, y, w, h, dimOrder, transform = None):   # transform의 기본값은 None으로 설정.
		"""
		Creates a new window with the specified dimensions and transform
		"""
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.dimOrder = dimOrder
		self.transform = transform
	
	def apply(self, matrix):
		"""
		Slices the supplied matrix and applies any transform bound to this window
		"""
		view = matrix[ self.indices() ]   # matrix는 뭐고, self.indices()는 뭐지?
		return self.transform(view) if self.transform != None else view   # transform이 주어지면 view에 적용하고, 없음 말고.
	
	def getRect(self):
		"""
		Returns the window bounds as a tuple of (x,y,w,h)
		"""
		return (self.x, self.y, self.w, self.h)   # tuple을 반환.
	
	def setRect(self, rect):    # what is rect?
		"""
		Sets the window bounds from a tuple of (x,y,w,h)
		"""
		self.x, self.y, self.w, self.h = rect   # rect(tuple)로부터 x, y, w, h 정보 끄집어냄
	
	def indices(self, includeChannel=True):
		"""
		Retrieves the indices for this window as a tuple of slices
    window의 indeces를 slice라는 object가 나열된 tuple로 얻는 함수. 이게 몬소리지..
		"""
		if self.dimOrder == DimOrder.HeightWidthChannel:
			
			# Equivalent to [self.y:self.y+self.h+1, self.x:self.x+self.w+1]
			return (                            # tuple
				slice(self.y, self.y+self.h),     # self.y부터 self.y+self.h-1까지의 인덱스를 지정.
				slice(self.x, self.x+self.w)
			)
			
		elif self.dimOrder == DimOrder.ChannelHeightWidth:
			
			if includeChannel is True:
				
				# Equivalent to [:, self.y:self.y+self.h+1, self.x:self.x+self.w+1]
				return (
					slice(None, None),
					slice(self.y, self.y+self.h),   # 한 window의 y좌표상 시작과 끝. self.y는 출발좌표, h는 window의 height
					slice(self.x, self.x+self.w)    # 즉 x시작, y시작, x끝, y끝 알려주고 '잘라라'(slice).. 로 추측중.
				)
				
			else:
				
				# Equivalent to [self.y:self.y+self.h+1, self.x:self.x+self.w+1]
				return (
					slice(self.y, self.y+self.h),
					slice(self.x, self.x+self.w)
				)
			
		else:
			raise Error('Unsupported order of dimensions: ' + str(self.dimOrder))
		
	def __str__(self):    # "x값, y값, w값, h값" 이라는 string을 반환.
		return '(' + str(self.x) + ',' + str(self.y) + ',' + str(self.w) + ',' + str(self.h) + ')'
	
	def __repr__(self):   # 이건 대체 왜있어야하지?
		return self.__str__()


def generate(data, dimOrder, maxWindowSize, overlapPercent, transforms=[], overrideWidth=None, overrideHeight=None):
	"""
	Generates a set of sliding windows for the specified dataset. 
  이미지가 저장된 형태(hwc 또는 chw)가 달라도 width와 height를 받기 위해서 지정한 함수.
	"""
	
	# Determine the dimensions of the input data
	width = data.shape[dimOrder.index('w')]
	height = data.shape[dimOrder.index('h')]
	
	# Generate the windows
	return generateForSize(width, height, dimOrder, maxWindowSize, overlapPercent, transforms, overrideWidth, overrideHeight)


def generateForSize(width, height, dimOrder, maxWindowSize, overlapPercent, transforms=[], overrideWidth=None, overrideHeight=None):
	"""
	Generates a set of sliding windows for a dataset with the specified dimensions and order.
  (특별한 사유가 없는 한) 주어진 값을 변의 길이로 하는 정사각형 window 만들어보자.
	"""
	
	# Create square windows unless an explicit width or height has been specified
	windowSizeX = maxWindowSize if overrideWidth is None else overrideWidth     # windowSizeX는, 만약 overrideWidth=None이면 maxWindowSize, None이 아니라 값이 주어져 있으면 그 주어진 overrideWidth.
	windowSizeY = maxWindowSize if overrideHeight is None else overrideHeight   # 만약 여기서 걸리는 특수한 상황이 아니라면, generateForSize 함수에 의해 생성되는 window는 가로, 세로가 모두 maxWindowSize인 square.
	
	# If the input data is smaller than the specified window size,
	# clip the window size to the input size on both dimensions
	windowSizeX = min(windowSizeX, width)   # 이미지가 window보다 작은 경우 발생할 에러 방지
	windowSizeY = min(windowSizeY, height)
	
	# Compute the window overlap and step size
	windowOverlapX = int(math.floor(windowSizeX * overlapPercent))    # math.floor; 버림    # x방향으로 window가 겹치는 길이.
	windowOverlapY = int(math.floor(windowSizeY * overlapPercent))
	stepSizeX = windowSizeX - windowOverlapX    # window가 x방향으로 한 번에 움직이는 길이(픽셀수).
	stepSizeY = windowSizeY - windowOverlapY
	
	# Determine how many windows we will need in order to cover the input data
	lastX = width - windowSizeX   # x방향으로 마지막 window가 시작하는 x좌표
	lastY = height - windowSizeY
	xOffsets = list(range(0, lastX+1, stepSizeX))   # 0부터 stepSizeX 간격으로 lastX+1 직전까지. lastX+1은 list에 포함되지 않음. 즉 +1은 lastX까지 list에 포함시켜 주기 위한 장치.
	yOffsets = list(range(0, lastY+1, stepSizeY))   # window가 시작되는 y좌표를 나열한 list. 즉 이 list의 길이는 window의 y방향 개수가 됨.
	
	# Unless the input data dimensions are exact multiples of the step size,
	# we will need one additional row and column of windows to get 100% coverage    # 즉 그림을 남기지 말고 window를 남기자.
	if len(xOffsets) == 0 or xOffsets[-1] != lastX:   # 도대체 이게 무슨 경우지? 모르겠지만 일단 넘어가자. error방지 코드겠지뭐.
		xOffsets.append(lastX)
	if len(yOffsets) == 0 or yOffsets[-1] != lastY:
		yOffsets.append(lastY)
	
	# Generate the list of windows
	windows = []
	for xOffset in xOffsets:
		for yOffset in yOffsets:    # 이 for loop의 순서대로라면 세로로 움직이는 window가 되지않나?
			for transform in [None] + transforms:   # 첫 번째 transform은 None이 된다.
				windows.append(SlidingWindow(   # SlidingWindow라는, 각 window의 크기와 위치 정보 포함된, class? 객체? 아무튼 이걸 차곡차곡 쌓는다.
					x=xOffset,    # 어떤 window가 시작되는 x좌표
					y=yOffset,    # 어떤 window가 시작되는 y좌표
					w=windowSizeX,    # window의 x방향 크기
					h=windowSizeY,    # window의 y방향 크기
					dimOrder=dimOrder,    # chw or hwc. 채널이 먼저 표시되어있냐, 나중에 표시되어있냐.
					transform=transform   # what is it?? 기본값은 None.
				))
	
	return windows


def generateRectanglarWindows(data, dimOrder, windowShape, overlapPercent, transforms=[]):
	"""
	Generates a set of sliding windows for the specified dataset, creating rectangular windows instead of square windows.
	`windowShape` must be a tuple specifying the desired window dimensions in (height,width) form.
  주어진 tuple을 높이, 너비로 하는 직사각형 window 만들어보자.
	"""
	
	# Determine the dimensions of the input data
	width = data.shape[dimOrder.index('w')]   # 여기서 width, height은 window가 아니라 input 이미지의 너비와 높이
	height = data.shape[dimOrder.index('h')]
	
	# Generate the windows
	windowHeight, windowWidth = windowShape   # tuple안에 저장된 값을 뽑아서 windowHeight, windowWidth에 지정함.
	return generateForSize(     # 위에서 만든 square window 함수를 활용.
		width,
		height,
		dimOrder,
		0,    # maxwindowsize. 즉 overrideWidth 혹은 overrideHeight가 지정되지 않으면 window의 너비 혹은 높이가 0이 되어버림. 반드시 windowWidth와 windowHeight를 지정하라는 의미.
		overlapPercent,
		transforms,
		overrideWidth = windowWidth,
		overrideHeight = windowHeight
	)


def generateForNumberOfWindows(data, dimOrder, windowCount, overlapPercent, transforms=[]):
	"""
	Generates a set of sliding windows for the specified dataset, automatically determining the required window size in
	order to create the specified number of windows. `windowCount` must be a tuple specifying the desired number of windows
	along the Y and X axes, in the form (countY, countX).
  이번에는 window의 크기를 지정하지 않고, 대신 개수를 지정해 주었을 경우에 직사각형 window를 구해보자!
	"""
	
	# Determine the dimensions of the input data
	width = data.shape[dimOrder.index('w')]   # data.shape의 세 가지 값 중 인덱스가 dimOrder.index('w')인 값을 width로 받음.
	height = data.shape[dimOrder.index('h')]  # 이 때 dimOrder.index('w')란, dimOrder에서 원소 'w'를 찾아 해당하는 인덱스를 반환.
	
	# Determine the window size required to most closely match the desired window count along both axes
	countY, countX = windowCount    # Y방향, X방향 원하는 window의 개수를 windowCount라는 튜플로 입력해 줬음. 그 tuple(희망사항)을 countY, countX로 일단 받음.
	windowSizeX = math.ceil(width / countX)   # math.ceil; 올림. 즉 windowSizeX는 가장 낭비가 없으면서도, (그림이 아니라) window를 남겼을 때의 x방향 window 크기로 정한다.
	windowSizeY = math.ceil(height / countY)
	
	# Generate the windows
	return generateForSize(
		width,
		height,
		dimOrder,
		0,    # maxWindowSize.
		overlapPercent,
		transforms,
		overrideWidth = windowSizeX,
		overrideHeight = windowSizeY
	)
