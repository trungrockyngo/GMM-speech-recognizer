import GMM_MultiDim as GMMD

def main():
	# 1-D GMM
	N = 100
	D = 1
	data = np.arange(-2,8.5,0.1)
	one_D_GMM = GMMD(N, D)
	# pl.plot(x,y,'k',linewidth=4)

	pl.figure()
	pl.plot(ll,'ko-')
	pl.show()

def speech_Testing():
	ah_features = np.load(“air_000_AH.feat”)
	s_features = np.load(“air_000_S.feat”)
	print ah_features.shape, s_features.shape

# if __name__ == "__main__":
# main()
speech_Testing()
