def augmentor(img):
    """
    Fixed augmentor function with proper numpy array handling
    """
    # Ensure input is a numpy array
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    
    # Ensure it's the right data type
    img = img.astype(np.uint8)
    
    TH, TW = img.shape

    param_gamma_low = 0.3
    param_gamma_high = 2
    param_mean_gaussian_noise = 0
    param_sigma_gaussian_noise = 100**0.5
    param_kanungo_alpha = 2
    param_kanungo_beta = 2
    param_kanungo_alpha0 = 1
    param_kanungo_beta0 = 1
    param_kanungo_mu = 0
    param_kanungo_k = 2
    param_min_shear = -0.5
    param_max_shear = 0.25
    param_rotation = 3
    param_scale = 0.2
    param_movement_BB = 6

    # Add gaussian noise
    gauss = np.random.normal(param_mean_gaussian_noise, param_sigma_gaussian_noise, (TH, TW))
    gauss = gauss.reshape(TH, TW)
    gaussiannoise = np.uint8(np.clip(np.float32(img) + gauss, 0, 255))

    # Ensure gaussiannoise is a proper numpy array with correct dtype
    gaussiannoise = np.asarray(gaussiannoise, dtype=np.uint8)

    # Randomly erode, dilate or nothing
    kernel = np.ones((3, 3), np.uint8)
    a = random.choice([2, 3])
    
    if a == 1:
        gaussiannoise = cv2.dilate(gaussiannoise, kernel, iterations=1)
    elif a == 2:
        gaussiannoise = cv2.erode(gaussiannoise, kernel, iterations=1)

    # Add random gamma correction
    gamma = np.random.uniform(param_gamma_low, param_gamma_high)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gammacorrected = cv2.LUT(np.uint8(gaussiannoise), table)

    # Binarize image with Otsu
    otsu_th, binarized = cv2.threshold(gammacorrected, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Kanungo noise
    try:
        dist = cv2.distanceTransform(1 - binarized, cv2.DIST_L1, 3)
        dist2 = cv2.distanceTransform(binarized, cv2.DIST_L1, 3)
    except:
        # Fallback if DIST_L1 doesn't work
        dist = cv2.distanceTransform(1 - binarized, cv2.DIST_L2, 3)
        dist2 = cv2.distanceTransform(binarized, cv2.DIST_L2, 3)

    dist = dist.astype('float64')
    dist2 = dist2.astype('float64')

    P = (param_kanungo_alpha0 * np.exp(-param_kanungo_alpha * dist**2)) + param_kanungo_mu
    P2 = (param_kanungo_beta0 * np.exp(-param_kanungo_beta * dist2**2)) + param_kanungo_mu
    distorted = binarized.copy()
    distorted[((P > np.random.rand(P.shape[0], P.shape[1])) & (binarized == 0))] = 1
    distorted[((P2 > np.random.rand(P.shape[0], P.shape[1])) & (binarized == 1))] = 0
    closing = cv2.morphologyEx(distorted, cv2.MORPH_CLOSE, np.ones((param_kanungo_k, param_kanungo_k), dtype=np.uint8))

    # Apply binary image as mask and put it on a larger canvas
    pseudo_binarized = closing * (255 - gammacorrected)
    canvas = np.zeros((3 * TH, 3 * TW), dtype=np.uint8)
    canvas[TH:2 * TH, TW:2 * TW] = pseudo_binarized
    points = []
    count = 0

    while len(points) < 1:
        count += 1
        if count > 50:
            break

        # Random shear
        shear_angle = np.random.uniform(param_min_shear, param_max_shear)
        M = np.float32([[1, shear_angle, 0], [0, 1, 0]])
        sheared = cv2.warpAffine(canvas, M, (3 * TW, 3 * TH), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)

        # Random rotation
        M = cv2.getRotationMatrix2D((3 * TW / 2, 3 * TH / 2), np.random.uniform(-param_rotation, param_rotation), 1)
        rotated = cv2.warpAffine(sheared, M, (3 * TW, 3 * TH), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)

        # Random scaling
        scaling_factor = np.random.uniform(1 - param_scale, 1 + param_scale)
        scaled = cv2.resize(rotated, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)

        # Detect cropping parameters
        points = np.argwhere(scaled != 0)
        points = np.fliplr(points)

    if len(points) < 1:
        return pseudo_binarized

    r = cv2.boundingRect(np.array([points]))

    # Random cropping
    deltax = random.randint(-param_movement_BB, param_movement_BB)
    deltay = random.randint(-param_movement_BB, param_movement_BB)
    x1 = min(scaled.shape[0] - 1, max(0, r[1] + deltax))
    y1 = min(scaled.shape[1] - 1, max(0, r[0] + deltay))
    x2 = min(scaled.shape[0], x1 + r[3])
    y2 = min(scaled.shape[1], y1 + r[2])
    final_image = np.uint8(scaled[x1:x2, y1:y2])

    return final_image