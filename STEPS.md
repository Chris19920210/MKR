1. Add (u, v, h, r, t) for fead_dict
```
original results:
epoch 0    train auc: 0.8892  acc: 0.8023    eval auc: 0.8837  acc: 0.8001    test auc: 0.8840  acc: 0.8007
epoch 1    train auc: 0.8958  acc: 0.8116    eval auc: 0.8862  acc: 0.8052    test auc: 0.8868  acc: 0.8057
epoch 2    train auc: 0.9027  acc: 0.8171    eval auc: 0.8884  acc: 0.8077    test auc: 0.8888  acc: 0.8075
epoch 3    train auc: 0.9122  acc: 0.8266    eval auc: 0.8937  acc: 0.8114    test auc: 0.8932  acc: 0.8118
epoch 4    train auc: 0.9200  acc: 0.8375    eval auc: 0.8988  acc: 0.8206    test auc: 0.8985  acc: 0.8192
epoch 5    train auc: 0.9254  acc: 0.8444    eval auc: 0.9034  acc: 0.8261    test auc: 0.9033  acc: 0.8249
epoch 6    train auc: 0.9289  acc: 0.8495    eval auc: 0.9062  acc: 0.8296    test auc: 0.9061  acc: 0.8284
epoch 7    train auc: 0.9314  acc: 0.8521    eval auc: 0.9081  acc: 0.8319    test auc: 0.9082  acc: 0.8308
epoch 8    train auc: 0.9328  acc: 0.8539    eval auc: 0.9095  acc: 0.8333    test auc: 0.9097  acc: 0.8328
epoch 9    train auc: 0.9344  acc: 0.8559    eval auc: 0.9102  acc: 0.8343    test auc: 0.9102  acc: 0.8338
epoch 10    train auc: 0.9355  acc: 0.8578    eval auc: 0.9112  acc: 0.8357    test auc: 0.9111  acc: 0.8350
epoch 11    train auc: 0.9360  acc: 0.8580    eval auc: 0.9119  acc: 0.8356    test auc: 0.9117  acc: 0.8352
epoch 12    train auc: 0.9369  acc: 0.8597    eval auc: 0.9113  acc: 0.8365    test auc: 0.9115  acc: 0.8354
epoch 13    train auc: 0.9372  acc: 0.8594    eval auc: 0.9120  acc: 0.8361    test auc: 0.9116  acc: 0.8348
epoch 14    train auc: 0.9377  acc: 0.8596    eval auc: 0.9122  acc: 0.8366    test auc: 0.9119  acc: 0.8357
epoch 15    train auc: 0.9377  acc: 0.8603    eval auc: 0.9121  acc: 0.8368    test auc: 0.9117  acc: 0.8362
epoch 16    train auc: 0.9376  acc: 0.8599    eval auc: 0.9126  acc: 0.8376    test auc: 0.9124  acc: 0.8364
epoch 17    train auc: 0.9385  acc: 0.8613    eval auc: 0.9126  acc: 0.8373    test auc: 0.9123  acc: 0.8359
epoch 18    train auc: 0.9386  acc: 0.8609    eval auc: 0.9122  acc: 0.8368    test auc: 0.9124  acc: 0.8369
epoch 19    train auc: 0.9389  acc: 0.8617    eval auc: 0.9128  acc: 0.8372    test auc: 0.9128  acc: 0.8373

new results:
```


2. Replace CrossCompress op with CycleGAN
3. Apply different type of KGE
4. Apply this model on other kinds of side information like social networks