#-*- coding: utf-8 -*-

# Matrix Factorization
# 코드 출처: http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/
# 위 사이트 코드를 보고 라인별 의미를 주석으로 추가




import numpy



def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T

    for step in xrange(steps): #iteration


        #모든 셀에 대에서 계산하지만 보통은 일부에 대해서 error를 계산한 후 gradient descent -> SGD (Stochastic Gradient Descent)
            #속도 향상
            #Overfitting 방지

        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0: #0은 값이 존재하지 않는 셀이기 때문에 고려하지 않음

                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j]) #error 계산

                    #R[i][j]와 관련된 P,Q의 셀을 업데이트
                        #관련된: R[i][j]를 계산하는데 필요한 셀
                        #업데이트: R[i][j]와 계산된 값의 오차를 이용 (gradient descent)

                    #업데이트
                        #alpha * (2 * eij * Q[k][j] - beta * P[i][k]): 미분식 (P[i][k]가 변할 때, eij의 변화량)
                            #alpha: gradient descent의 방향을 어느정도 이동할지 정하지는 변수
                        #beta * P[i][k]: overfitting을 막기 위한 Regularization 식
                            #P[i][k]의 값을 error로 간주하여, 큰 값이 들어가지 않게 막음 -> 큰 값이 들어가면 error가 커짐
                            #beta: Regularization할 때 P[i][k]를 얼마나 error로 간주할 것인지 결정하는 변수 
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)

        #error 계산: 종료 조건을 체크하기 위해 (mymedialite는 이 부분 없이 iteration 횟수로 종료)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))

        if e < 0.001:
            break

    return P, Q.T





#input 행렬
    # 행을 유저, 열을 아이템으로 가정하고 아래 내용 설명)
    # 0은 값이 존재하지 않는 셀
R = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]

R = numpy.array(R) #numpy array로 행렬 변환
N = len(R) #유저 수 5
M = len(R[0]) #아이템 수 4
K = 2 #latent factor 수 (=차원 or d or f)

P = numpy.random.rand(N,K) #유저 행렬 초기화 (5x2)
Q = numpy.random.rand(M,K) #아이템 행렬 초기화 (4x2)
print P
print Q.T



#Input
#R: input 행렬: 오차 계산에 이용
#P: 초기화된 유저 행렬
#Q: 초기화된 아이템 행렬
#K: latent factor 수 (=차원 or d or f)

#Output
#nP: 최종적으로 학습된 유저 행렬
#nQ: 최종적으로 학습된 아이템 행렬
nP, nQ = matrix_factorization(R, P, Q, K)


nR = numpy.dot(nP, nQ.T) #최종 output 계산
print nR
