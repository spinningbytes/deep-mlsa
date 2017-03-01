def evaluate(model, test_iteraotr, experiment_name, ofname):
    inputs = test_iteraotr.input_data
    outputs = test_iteraotr.output_data
    names = test_iteraotr.names

    for i, o, n in zip(inputs, outputs, names):
        ofile = open(ofname, 'wt')

        sentence_embeddings = model.predict(i)

        for list in sentence_embeddings:
            slist = [str(x) for x in list.tolist()]

            output_line = '{}\n'.format(' '.join(slist))
            ofile.write(output_line)

    return output_line