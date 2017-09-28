from utils import run_utils
from sklearn.cross_validation import KFold
from os.path import join


def fit_model(config_data, model, train_iterator, valid_iterator=None):
    if not config_data.get('cross_valid'):
        return fit_model_single(config_data, model, train_iterator, valid_iterator)
    elif config_data.get('cross_valid') == 'true':
        return fit_model_cv(config_data, model, train_iterator, valid_iterator)
    else:
        return fit_model_single(config_data, model, train_iterator, valid_iterator)


def fit_model_cv(config_data, model, train_iterator, valid_iterator):
    assert train_iterator.type == 'loader'

    nb_epochs = config_data['nb_epochs']
    batch_size = config_data['batch_size']

    X_train = train_iterator.input_data
    y_train = train_iterator.output_data

    kf = KFold(n_folds=5, shuffle=True, n=len(X_train[0]))

    path = config_data['output_path']
    basename = config_data['output_basename']
    base_path = join(path, basename)
    opath = join(base_path, 'base_model.h5')
    model.save_weights(opath)

    appendices = []

    for i, (train, test) in enumerate(kf):
        model.load_weights(opath)

        input_train = [X[train] for X in X_train]
        output_train = [y[train] for y in y_train]

        input_valid = [X[test] for X in X_train]
        output_valid = [y[test] for y in y_train]

        appendix = '_{}'.format(i)
        callbacks = run_utils.get_callbacks(config_data, appendix=appendix)
        stored_model = True
        hist = model.fit(
            x=input_train,
            y=output_train,
            batch_size=batch_size,
            validation_data=(input_valid, output_valid),
            nb_epoch=nb_epochs,
            verbose=1,
            callbacks=callbacks,
            class_weight=run_utils.get_classweight(config_data)
        )

        appendices.append(appendix)

        weights_path = join(base_path, 'best_model{}.h5'.format(appendix))
        model.load_weights(weights_path)

        oline_test = run_utils.get_evaluation(config_data, model, train_iterator, basename, '')
        print(oline_test)

    return hist, stored_model, appendices


def fit_model_single(config_data, model, train_iterator, valid_iterator):
    stored_model = False
    nb_epochs = config_data['nb_epochs']
    batch_size = config_data['batch_size']

    if not train_iterator:
        return None, False, ''

    if train_iterator.type == 'generator' and train_iterator.nsamples > 0:
        callbacks, stored_model = run_utils.get_callbacks(config_data)
        if valid_iterator:
            if valid_iterator.type == 'loader':
                hist = model.fit_generator(
                    train_iterator.flow(batch_size),
                    samples_per_epoch=train_iterator.nsamples,
                    validation_data=(valid_iterator.input_data, valid_iterator.output_data),
                    nb_epoch=nb_epochs,
                    verbose=1,
                    callbacks=callbacks
                )
            elif valid_iterator.type == 'generator':
                callbacks, stored_model = run_utils.get_callbacks(config_data)
                hist = model.fit_generator(
                    train_iterator.flow(batch_size),
                    samples_per_epoch=train_iterator.nsamples,
                    validation_data=valid_iterator.flow(batch_size),
                    nb_val_samples=valid_iterator.nsamples,
                    nb_epoch=nb_epochs,
                    verbose=1,
                    callbacks=callbacks
                )
        else:
            callbacks, stored_model = run_utils.get_callbacks(config_data)
            hist = model.fit_generator(
                train_iterator.flow(batch_size),
                samples_per_epoch=train_iterator.nsamples,
                nb_epoch=nb_epochs,
                verbose=1,
                callbacks=callbacks
            )
    elif train_iterator.type == 'loader':
        if valid_iterator:
            if valid_iterator.type == 'loader':
                callbacks, stored_model = run_utils.get_callbacks(config_data)
                stored_model = True
                hist = model.fit(
                    x=train_iterator.input_data,
                    y=train_iterator.output_data,
                    batch_size=batch_size,
                    validation_data=(valid_iterator.input_data, valid_iterator.output_data),
                    nb_epoch=nb_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    class_weight=run_utils.get_classweight(config_data)
                )
            elif valid_iterator.type == 'generator':
                callbacks, stored_model = run_utils.get_callbacks(config_data)
                hist = model.fit(
                    x=train_iterator.input_data,
                    y=train_iterator.output_data,
                    batch_size=batch_size,
                    validation_data=valid_iterator.flow(batch_size),
                    nb_epoch=nb_epochs,
                    verbose=1,
                    callbacks=callbacks

                )
        else:
            callbacks, stored_model = run_utils.get_callbacks(config_data)
            hist = model.fit(
                x=train_iterator.input_data,
                y=train_iterator.output_data,
                batch_size=batch_size,
                nb_epoch=nb_epochs,
                verbose=1,
                callbacks=callbacks
            )

    return hist, stored_model, ''


