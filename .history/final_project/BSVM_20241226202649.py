Traceback (most recent call last):
  File "c:\Users\143\OneDrive\桌面\2024秋季\人工智能与机器学习\final_project\BSVM.py", line 104, in <module> 
    Z_linear_pred = linear_svc.predict(np.c_[xx.ravel(), yy.ravel()])
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\143\OneDrive\桌面\2024秋季\人工智能与机器学习\final_project\lec11_svm_code\lec11_svm\svc.py", line 80, in predict
    return (self.decision_function(np.array(X)) >= 0).astype(int)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\143\OneDrive\桌面\2024秋季\人工智能与机器学习\final_project\lec11_svm_code\lec11_svm\svc.py", line 76, in decision_function
    return np.matmul(self.coef_[0], np.array(X).T) - self.coef_[-1]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 2 is different from 7)