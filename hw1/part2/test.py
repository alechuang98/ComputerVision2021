import numpy as np
import time

def test():
    a = np.random.randint(256, size=(1000000, 3))
    dic = {}
    arr = np.zeros((256, 256, 256))
    arr2 = np.zeros((256))

    for i in range(256):
        arr2[i] = np.exp(- (i ** 2) / (2 * 0.1 ** 2 * 255 ** 2))
        for j in range(256):
            for k in range(256):
                arr[i][j][k] = np.exp(- (i ** 2 + j ** 2 + k ** 2) / (2 * 0.1 ** 2 * 255 ** 2))

    for i in range(256):
        for j in range(i, 256):
            for k in range(j, 256):
                dic[i ** 2 + j ** 2 + k ** 2] = np.exp(- (i ** 2 + j ** 2 + k ** 2) / (2 * 0.1 ** 2 * 255 ** 2))

    def get_dic():
        ans = 0
        for i in range(a.shape[0]):
            ans += dic[np.sum(a[i] ** 2)]
        return ans

    def naive():
        ans = 0
        for i in range(a.shape[0]):
            ans += np.exp(- (a[i][0] ** 2 + a[i][1] ** 2 + a[i][2] ** 2) / (2 * 0.1 ** 2 * 255 ** 2))
        return ans

    def get_array():
        ans = 0
        for i in range(a.shape[0]):
            ans += arr[a[i][0], a[i][1], a[i][2]]
        return ans
    
    def get_array2():
        ans = 0
        for i in range(a.shape[0]):
            ans += arr2[a[i][0]] * arr2[a[i][1]] * arr2[a[i][2]]
        return ans

    print('build ok')
    t0 = time.time()
    k = get_array()
    print(k)
    print('time %.4f s' % (time.time() - t0))
    t0 = time.time()
    k = get_array2()
    print(k)
    print('time %.4f s' % (time.time() - t0))


test()
