./run 1 1000 random deep random_d10_deep 1 mean 10 0
./run 1 1000 random shallow random_d10_shallow 1 mean 10 0
./run 1 1000 lookup shallow lookup_d10_shallow 1 mean 10 0

./run 1 1000 sentence deep sentence_d10_deep_mean 1 mean 10 0
./run 1 1000 sentence shallow sentence_d10_shallow_mean 1 mean 10 0
./run 1 1000 sentence deep sentence_d10_deep_prod 1 prod 10 0
./run 1 1000 sentence shallow sentence_d10_shallow_prod 1 prod 10 0

./run 1 1000 bow deep bow_deep 1 mean 10 0
./run 1 1000 bow shallow bow_shallow 1 mean 10 0

./run 1 1000 bob deep bob_deep 1 mean 10 0
./run 1 1000 bob shallow bob_shallow 1 mean 10 0

./run 1 1000 lstm deep lstm_deep 1 mean 20 0
./run 1 1000 lstm shallow lstm_shallow 1 mean 20 0

./run 1 1000 mvrnn deep mvrnn_d10_MEAN 1 mean 10 0
./run 1 1000 mvrnn shallow mvrrn_d10_MEAN 1 mean 10 0

./run 1 1000 mvrnn deep mvrnn_d10_PROD 1 prod 10 0
./run 1 1000 mvrnn shallow mvrrn_d10_PROD 1 prod 10 0