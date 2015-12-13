require 'nn'
-- require 'cunn'
require 'torch'
require 'dpnn'
require 'mvrnn'
require 'parse'
require 'text_to_vector'
require 'add_diagonal_matrix'
require 'nntools'

local clone_net = nntools.clone_net


local function make_vector_networks (tree, word_vecs, W, non_linearity, predictor)

    local vector_networks = {}
    local prediction_networks = {}

    local function f(tree)
        if tree.terminal then
            return clone_net(word_vecs[tree.word])
        end

        local left_vector, right_vector, concat, network

        left_vector = f(tree.left, false)
        right_vector = f(tree.right, false)

        concat = nn.Concat(1):add(left_vector):add(right_vector)

        network = nn.Sequential()
        network:add(concat)
        network:add(clone_net(W))
        network:add(non_linearity:clone())

        vector_networks[tree] = network

        if predictor then
            prediction_networks[tree] = (
                nn.Sequential()
                :add(network)
                :add(clone_net(predictor)))
        end

        return network
    end

    f(tree)
    return vector_networks, prediction_networks
end

do
    local RecNN, parent = torch.class('RecNN', 'MVRNN')

    function RecNN:__init(
             n, words, non_linearity, predictor, repr_predictor, repr_criterion,
             combine_mode, wv_func, allow_new_words, max_repr_train)

        -- combine_mode is one of MEAN, SUM, CAT, PROD
        -- wv_func is an optional function for setting initial word vectors

        text_to_vector.Text2Vector.__init(self, n, false)

        self.predictor = predictor

        self.repr_predictor = repr_predictor
        self.repr_criterion = repr_criterion or nn.MSECriterion()

        -- Maximum number of representation training steps to do per call to accGradReprParameters
        self.max_repr_train = max_repr_train or 0

        self.wv_func = wv_func

        self.non_linearity = non_linearity or nn.Tanh()
        self.combine_mode = combine_mode or MEAN
        self.allow_new_words = allow_new_words or false

        print("N: ", type(n))

        -- Create W
        self.W = nn.Linear(2*n, n)

        self.word_vecs = {}
        self.w_shape = torch.LongStorage{n}

        -- A mapping: text fragment -> (sentence -> parse tree)
        self.parse_trees = {}

        self.vector_networks = {}
        self.prediction_networks = {}
        self.forward_networks = {}

        -- words must be a set
        for word, _ in pairs(words) do
            self:add_word(word)
        end
    end

    function RecNN:updateOutput(text)
        local batch_mode, size, pts, txt_i, concat, forward_net, result
        batch_mode, size, text = self:handleBatch(text)

        self.output = nil

        for i=1,size do
            pts, txt_i = self:parse(text[i])

            forward_net = self.forward_networks[txt_i]

            if not forward_net then
                concat = nn.ConcatTable()
                for s, tree in pairs(pts) do
                    concat:add(
                        nn.Sequential()
                        :add(self.vector_networks[tree])
                        :add(nn.Reshape(self.dimension, 1)))
                end

                forward_net = nn.Sequential():add(concat)

                if self.combine_mode == mvrnn.MEAN then
                    forward_net:add(nn.JoinTable(2)):add(nn.Mean(2))
                elseif self.combine_mode == mvrnn.SUM then
                    forward_net:add(nn.JoinTable(2)):add(nn.Sum(2))
                elseif self.combine_mode == mvrnn.CAT then
                    forward_net:add(nn.JoinTable(1))
                elseif self.combine_mode == mvrnn.PROD then
                    forward_net:add(nn.CMulTable())
                else
                    error("Invalid value for combine_mode: " .. self.combine_mode)
                end

                forward_net:add(clone_net(self.predictor))
                self.forward_networks[txt_i] = forward_net
            end

            result = forward_net:forward(torch.Tensor())

            if i == 1 then
                is_table = type(result) == 'table'
                self.output = {}
            end

            if not is_table then
                result = {result}
            end

            for i=1,#result do
                reshaped = result[i]:reshape(1, result[i]:size(1))
                if not self.output[i] then
                    self.output[i] = reshaped
                else
                    self.output[i] = torch.cat(self.output[i], reshaped, 1)
                end
            end
        end

        if not is_table then
            self.output = self.output[1]
        end

        if not batch_mode then
            self.output = self.output:reshape(self.output:size(2))
        end

        return self.output
    end

    -- Add a new word, and create a new matrix and vector for it.
    function RecNN:add_word(word)
        if not self.word_vecs[word] then
            local u_shape, v_shape, a_shape, w_shape = self.u_shape, self.v_shape, self.a_shape, self.w_shape

            w_net = (
                nn.Sequential()
                :add(nn.Constant(torch.zeros(w_shape)))
                :add(nn.Add(w_shape)))

            w_net:get(1).size = w_shape -- Fact that we need to do this seems like a bug
            w_net:get(2).bias = self.wv_func(w_shape, word)

            self.word_vecs[word] = w_net
        end
    end

    -- Parse a text fragment, create corresponding networks.
    -- ``text`` is a string or string-like (e.g. ByteVector)
    -- Returns both an array of mappings from sentences to parse_trees,
    -- and the text after it has been preprocessed.
    function RecNN:parse(text)
        text = self:preprocessText(text)

        if not self.parse_trees[text] then
            local parse_trees, pts, words
            local w_net, vector_networks, prediction_networks

            parse_trees = parse.parse_sentences({text})

            -- Use the first parse.
            pts = {}
            for s, trees in pairs(parse_trees) do
                pts[s] = trees[1]
            end
            parse_trees = pts
            self.parse_trees[text] = parse_trees

            words = {}
            for s, tree in pairs(parse_trees) do
                for w, v in pairs(tree:get_words()) do
                    words[w] = v
                end
            end

            for word, _ in pairs(words) do
                if not self.word_vecs[word] then
                    if self.allow_new_words then
                        self:add_word(word)
                    else
                        error("Adding new word: " .. word)
                    end
                end
            end

            for s, tree in pairs(parse_trees) do
                if self.max_repr_train > 0 then
                    predictor = self.repr_predictor or self.predictor
                else
                    predictor = nil
                end

                vector_networks, prediction_networks = make_vector_networks(
                    tree, self.word_vecs, self.W, self.non_linearity, predictor)

                nntools.update_table(self.vector_networks, vector_networks)
                nntools.update_table(self.prediction_networks, prediction_networks)
            end
        end

        return self.parse_trees[text], text
    end

    -- Return all parameters as an array of tensors.
    function RecNN:parameters()
        local params, grad_params = {}, {}

        for word, word_vec in pairs(self.word_vecs) do
            params[#params + 1] = word_vec:get(2).bias
            grad_params[#grad_params + 1] = word_vec:get(2).gradBias
        end

        params[#params + 1] = self.W.bias
        grad_params[#grad_params + 1] = self.W.gradBias

        if self.repr_predictor then
            local pred_params, pred_grad_params = self.repr_predictor:parameters()

            for i, p in ipairs(pred_params) do
                params[#params + 1] = p
            end

            for i, p in ipairs(pred_grad_params) do
                grad_params[#grad_params + 1] = p
            end
        end

        pred_params, pred_grad_params = self.predictor:parameters()

        for i, p in ipairs(pred_params) do
            params[#params + 1] = p
        end

        for i, p in ipairs(pred_grad_params) do
            grad_params[#grad_params + 1] = p
        end

        return params, grad_params
    end
end

local function test_recnn_train(max_repr_train, do_manual, lambda)
    print("Testing RecNN...")
    n = 20
    non_linearity = nn.Tanh()

    wv_func = function (size) return torch.Tensor(size):normal(0, 1.0) end

    combine_mode = MEAN
    predictor = nn.Linear(n, 1)

    learning_rate = 0.03

    text = "The end is near. They won the game."
    text_target = torch.Tensor{1.0}

    batch = {
        "The end of the dog.",
        "We won a game.",
        "The end of the world is coming soon.",
        "The end of the world is not coming."}
    batch_target = torch.Tensor{{1.0, 2.0, 3.0, 4.0}}:t()

    criterion = nn.MSECriterion()

    repr_predictor = nn.Linear(n, 1)
    repr_criterion = nn.MSECriterion()

    -- Get all words.
    local all_text = table.concat(batch, ' ') .. " " .. text
    local all_words = mvrnn.get_all_words(all_text)

    local recnn = RecNN(
        n, all_words, non_linearity, predictor, repr_predictor, repr_criterion,
        combine_mode, wv_func, false, max_repr_train)

    -- Very important that this happens before any text is fed to the network
    local w, dw = recnn:getParameters()

    print("Parameter norm: ", torch.norm(w))
    print("Gradient norm: ", torch.norm(dw))

    init_val = recnn:forward(text)
    init_vals = recnn:forward(batch)
    not_before = recnn:get_word_vec('not'):clone()

    start = os.time()

    local max_steps = 1000
    local n_steps = 0
    local grad_norm = 1.0
    while grad_norm > 0.01 and n_steps < max_steps do
        recnn:zeroGradParameters()

        -- Single instance mode
        recnn:accGradReprParameters(text, text_target)

        criterion:forward(recnn:forward(text), text_target)
        recnn:backward(text, criterion:backward(recnn.output, text_target))

        -- Batch mode
        recnn:accGradReprParameters(batch, batch_target)

        criterion:forward(recnn:forward(batch), batch_target)
        recnn:backward(batch, criterion:backward(recnn.output, batch_target))

        if do_manual then
            dw:add(lambda, w)
            w:add(-learning_rate, dw)
        else
            recnn:updateParameters(learning_rate)
        end

        grad_norm = torch.norm(dw)

        print("AFTER UPDATING: ")
        print("Parameter norm: ", torch.norm(w))
        print("Gradient norm: ", torch.norm(dw))
        print("n_steps: ", n_steps)

        n_steps = n_steps + 1
        learning_rate = learning_rate - 0.00001
    end

    final_val = recnn:forward(text)
    final_vals = recnn:forward(batch)
    not_after = recnn:get_word_vec('not'):clone()

    finish = os.time()

    print("Training " .. n_steps .. " steps took " .. finish - start .. " seconds.")

    print("Initial: ")
    print(init_val)
    print(init_vals)
    print(not_before)

    print("Final: ")
    print(final_val)
    print(final_vals)
    print(not_after)

    print "Passed :-)."
end


if arg[1] == 'test_recnn' or arg[1] == 'test' then
    test_recnn_train(5, false, 0.0)
    test_recnn_train(0, false, 0.0)
    test_recnn_train(5, true, 0.0)
    test_recnn_train(0, true, 0.0)
end

recursive_nn = {
    RecNN=RecNN,
    make_recnn = function (
            n, words, non_linearity, predictor, repr_predictor,
            repr_criterion, combine_mode, wv_func, allow_new_words, max_repr_train)

        return RecNN(
            n, words, non_linearity, predictor, repr_predictor,
            repr_criterion, combine_mode, wv_func, allow_new_words, max_repr_train), n
    end,

    get_all_words=nntools.get_all_words,

    CAT=mvrnn.CAT,
    MEAN=mvrnn.MEAN,
    SUM=mvrnn.SUM,
    PROD=mvrnn.PROD
}

return recursive_nn