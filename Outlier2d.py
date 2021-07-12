def outlier2d(image, kernel_size, threshold, padding=0, strides=1):

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel_size
    yKernShape = kernel_size
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]
    
    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    #print(yImgShape)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    # Iterate through image
    for y in range(imagePadded.shape[1]):
        # Exit Convolution
        if y > imagePadded.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(imagePadded.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > imagePadded.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        centre = int((kernel_size -1 )/2)
                        median_value = np.median(imagePadded[x: x + xKernShape, y: y + yKernShape])
                        centre_pixel = imagePadded[x + centre, y + centre]
                        if abs(centre_pixel - median_value) > threshold:
                            output[x, y] = median_value
                        else:
                            output[x, y] = centre_pixel    
                except:
                    break
    return output    