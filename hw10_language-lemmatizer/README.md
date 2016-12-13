# HW-10 lemmatizer (2-6pts, due Dec 19)

Implement network performing lemmatization for Czech and English. Use the data from the previous task (http://ufal.mff.cuni.cz/~straka/courses/npfl114/2016/morpho_data.zip). Note that the lemmas are all in lowercase.

You should start with the labs09/lemmatizer-skeleton.py file.

This task has several subtasks, you can solve only some of them if you want. In every subtask, represent a form using concatenation of final states of bidirectional GRU run on the form's characters.

* individual_decoder (2 points): generate every lemma independently, using GRU as a decoder, producing one lemma letter at a time (use labs09/contrib_seq2seq.py as a dynamic rnn decoder, see labs09/rnn_example_decoder.py for a simple usage)
* individual_attention_decoder (2 point): as in individual_decoder, but use attention
* combined_attention_decoder (2 point): use the same approach as in the individual_attention_decoder, but use additional sentence-level bidirectional GRU (i.e., the form representations are processed by a bidirectional GRU and the results are used for the lemma generation)
* English competition (1-3): using any deep learning approach which uses only the data in the provided archive, try achieving highest accuracy on English testing data. The solution to this subtask is both a source code of you network and annotated testing data, which will be evaluated using the labs08/morpho_evaluate.py script. The points will be awarded according to the accuracy reached – three best submissions get 3 points, next three best submissions get 2 points and next three submissions get 1 point
* Czech competition (1-3): using any deep learning approach which uses only the data in the provided archive, try achieving highest accuracy on Czech testing data. The solution to this subtask is both a source code of you network and annotated testing data, which will be evaluated using the labs08/morpho_evaluate.py script. The points will be awarded according to the accuracy reached – three best submissions get 3 points, next three best submissions get 2 points and next three submissions get 1 point
