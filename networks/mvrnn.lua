require 'nn'
-- require 'cunn'
require 'torch'
require 'dpnn'
require 'parse'
require 'text_to_vector'
require 'add_diagonal_matrix'
require 'nntools'

-- Possible values for ``combine_mode``
local CAT = 'cat'
local SUM = 'sum'
local MEAN = 'mean'
local PROD = 'prod'

local clone_net = nntools.clone_net

local function make_matrix_networks (tree, U, V, A, WM)
    local matrix_networks = {}

    local function f (tree)
        local concat, network, left_network, right_network

        if tree.terminal then
            concat = nn.ConcatTable():add(clone_net(U[tree.word])):add(clone_net(V[tree.word]))
            network = nn.Sequential():add(concat):add(nn.MM()):add(A[tree.word])
        else
            left_network = f(tree.left)
            right_network = f(tree.right)

            concat = nn.Concat(1):add(left_network):add(right_network)
            network = nn.Sequential()
            network:add(nn.ConcatTable():add(clone_net(WM)):add(concat))
            network:add(nn.MM())
        end

        matrix_networks[tree] = network
        return network
    end

    f(tree)
    return matrix_networks
end


local function make_vector_networks (
        tree, matrix_networks, word_vecs, W, non_linearity, predictor)

    local vector_networks = {}
    local prediction_networks = {}

    local function f(tree)
        if tree.terminal then
            return clone_net(word_vecs[tree.word])
        end

        local left_vector, left_matrix, left_stream
        local right_vector, right_matrix, right_stream
        local concat, network

        left_vector = f(tree.left, false)
        left_matrix = clone_net(matrix_networks[tree.left])

        right_vector = f(tree.right, false)
        right_matrix = clone_net(matrix_networks[tree.right])

        left_stream = nn.Sequential()
        left_stream:add(
            nn.ConcatTable():add(right_matrix):add(left_vector))
        left_stream:add(nn.MM())

        right_stream = nn.Sequential()
        right_stream:add(
            nn.ConcatTable():add(left_matrix):add(right_vector))
        right_stream:add(nn.MM())

        concat = nn.Concat(1):add(left_stream):add(right_stream)

        network = nn.Sequential()
        network:add(nn.ConcatTable():add(clone_net(W)):add(concat))
        network:add(nn.MM())
        network:add(non_linearity:clone())

        vector_networks[tree] = network

        if predictor then
            prediction_networks[tree] = (
                nn.Sequential()
                :add(network)
                :add(nn.View(W:get(1).size[1]))
                :add(clone_net(predictor)))
        end

        return network
    end

    f(tree)
    return vector_networks, prediction_networks
end

do
    local MVRNN, parent = torch.class('MVRNN', 'Text2Vector')

    function MVRNN:__init(
             n, r, words, non_linearity, predictor, repr_predictor, repr_criterion,
             combine_mode, mat_func, a_func, wv_func, allow_new_words, max_repr_train)

        -- combine_mode is one of MEAN, SUM, CAT, PROD
        -- wv_func is an optional function for setting initial word vectors
        -- mat_func used to initialize U and V, a_func used to initialize A

        text_to_vector.Text2Vector.__init(self, n, false)

        self.predictor = predictor

        self.repr_predictor = repr_predictor
        self.repr_criterion = repr_criterion or nn.MSECriterion()

        -- Maximum number of representation training steps to do per call to accGradReprParameters
        self.max_repr_train = max_repr_train or 0

        self.mat_func = mat_func
        self.a_func = a_func
        self.wv_func = wv_func

        self.non_linearity = non_linearity or nn.Tanh()
        self.combine_mode = combine_mode or MEAN
        self.allow_new_words = allow_new_words or false

        -- Create W
        W_shape = torch.LongStorage{n, 2*n}
        W = nn.Sequential():add(nn.Constant(torch.zeros(W_shape))):add(nn.Add(W_shape))
        W:get(1).size = W_shape

        self.W = W

        -- Create WM
        WM_shape = torch.LongStorage{n, 2*n}
        WM = nn.Sequential():add(nn.Constant(torch.zeros(WM_shape))):add(nn.Add(WM_shape))
        WM:get(1).size = WM_shape

        self.WM = WM

        self.U = {}
        self.V = {}
        self.A = {}
        self.word_vecs = {}

        self.u_shape = torch.LongStorage{n, r}
        self.v_shape = torch.LongStorage{r, n}
        print("n, r: ", n, r)
        print("my v: ", self.v_shape)
        self.a_shape = n

        self.w_shape = torch.LongStorage{n, 1}

        -- A mapping: text fragment -> (sentence -> parse tree)
        self.parse_trees = {}

        self.matrix_networks = {}
        self.vector_networks = {}
        self.prediction_networks = {}
        self.forward_networks = {}

        -- words must be a set
        for word, _ in pairs(words) do
            self:add_word(word)
        end
    end

    function MVRNN:updateOutput(text)
        local batch_mode, size, pts, txt_i, concat, forward_net, result
        batch_mode, size, text = self:handleBatch(text)

        self.output = nil

        for i=1,size do
            pts, txt_i = self:parse(text[i])

            forward_net = self.forward_networks[txt_i]

            if not forward_net then
                concat = nn.ConcatTable()
                for s, tree in pairs(pts) do
                    concat:add(self.vector_networks[tree])
                end

                forward_net = nn.Sequential():add(concat)

                if self.combine_mode == MEAN then
                    forward_net:add(nn.JoinTable(2)):add(nn.Mean(2))
                elseif self.combine_mode == SUM then
                    forward_net:add(nn.JoinTable(2)):add(nn.Sum(2))
                elseif self.combine_mode == CAT then
                    forward_net:add(nn.JoinTable(1))
                elseif self.combine_mode == PROD then
                    forward_net:add(nn.CMulTable()):add(nn.Reshape(self.dimension))
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

    -- Handle batches. If we don't have a batch, turn it into a batch of size
    -- one so that it can be handled in the same way as a true batch.
    function MVRNN:handleBatch(text, target)
        local typename, size, batch_mode

        typename = torch.typename(text)
        if type(text) == 'table' or typename and text:dim() > 1 then
            -- Batch mode
            size = type(text) == 'table' and #text or text:size(1)
            if target then
                assert(
                    (type(target) == 'table' and #target == size)
                    or (torch.typename(target) and target:size(1) == size),
                    "Target count does not match input count.")
            end

            batch_mode = true

        elseif typename and text:dim() == 1 or type(text) == 'string' then
            size = 1
            text = {text}

            if target then
                target = {target}
            end

            batch_mode = false
        else
            error("Invalid input type. Expected vector, matrix, string or array of strings.")
        end

        return batch_mode, size, text, target
    end

    -- If in representation-training mode, then simply accumulate gradients for the predictor.
    -- Otherwise, accumulate gradients all the way down.
    function MVRNN:accGradParameters(text, gradOutput, scale)
        local batch_mode, size, predictor_input, pts, txt_i
        batch_mode, size, text, gradOutput = self:handleBatch(text, gradOutput)

        -- Train the entire network
        for i=1,size do
            pts, txt_i = self:parse(text[i])
            forward_net = self.forward_networks[txt_i]
            forward_net:backward(torch.Tensor(), gradOutput[i])
        end
    end

    -- Basically, this should be called in addition to calling backward if in representation training mode.
    -- Accumulates gradients for the representation.
    -- `mask` is optional, and can be used to set errors returned by the criterion to 0, so
    -- that only a subset of the outputs are actually trained.
    function MVRNN:accGradReprParameters(text, target, mask)
        local batch_mode, size, input, pts_i, trgt, n_sentences, constituent_subtrees, nets_to_train

        if not (self.max_repr_train > 0) then
            return
        end

        batch_mode, size, text, target = self:handleBatch(text, target)

        nets_to_train = {}

        -- Find all the subtrees that are affected
        for i=1,size do
            pts_i = self:parse(text[i])
            trgt = target[i]

            n_sentences = nntools.table_length(pts_i)

            for s, tree in pairs(pts_i) do
                constituent_subtrees = nntools.filter(
                    tree:get_subtrees(),
                    function (t, v) return not t.terminal end)

                for t, _ in pairs(constituent_subtrees) do
                    assert(self.prediction_networks[t], tostring(t))
                    nets_to_train[#nets_to_train+1] = {
                        self.prediction_networks[t], trgt, 1.0/(n_sentences * size), i}
                end
            end
        end

        if self.max_repr_train < #nets_to_train then
            local perm = torch.randperm(self.max_repr_train)
            local ntt = {}
            for i = 1, self.max_repr_train do
                ntt[i] = nets_to_train[perm[i]]
                ntt[i][3] = 1 / self.max_repr_train
            end
            nets_to_train = ntt
        end

        -- Accumulate gradients for those subtrees
        for i, net in ipairs(nets_to_train) do
            local prediction = net[1]:forward(torch.Tensor())
            local err = self.repr_criterion:forward(prediction, net[2])
            local gradInput = self.repr_criterion:backward(prediction, net[2])

            if mask then
                gradInput:cmul(mask[net[4]])
            end

            net[1]:backward(torch.Tensor(), gradInput, net[3])
        end
    end

    -- Add a new word, and create a new matrix and vector for it.
    function MVRNN:add_word(word)
        if not self.word_vecs[word] then
            local u_shape, v_shape, a_shape, w_shape = self.u_shape, self.v_shape, self.a_shape, self.w_shape

            w_net = (
                nn.Sequential()
                :add(nn.Constant(torch.zeros(w_shape)))
                :add(nn.Add(w_shape)))

            w_net:get(1).size = w_shape -- Fact that we need to do this seems like a bug
            w_net:get(2).bias = self.wv_func(w_shape, word)

            self.word_vecs[word] = w_net

            u_net = (
                nn.Sequential()
                :add(nn.Constant(torch.zeros(u_shape)))
                :add(nn.Add(u_shape)))

            u_net:get(1).size = u_shape
            u_net:get(2).bias = self.mat_func(u_shape)
            self.U[word] = u_net

            print("V: ", v_shape)
            v_net = (
                nn.Sequential()
                :add(nn.Constant(torch.zeros(v_shape)))
                :add(nn.Add(v_shape)))

            v_net:get(1).size = v_shape
            v_net:get(2).bias = self.mat_func(v_shape)
            self.V[word] = v_net

            a_net = AddDiagonalMatrix(a_shape)
            a_net.bias = self.a_func(a_shape)
            self.A[word] = a_net
        end
    end

    -- Parse a text fragment, create corresponding networks.
    -- ``text`` is a string or string-like (e.g. ByteVector)
    -- Returns both an array of mappings from sentences to parse_trees,
    -- and the text after it has been preprocessed.
    function MVRNN:parse(text)
        text = self:preprocessText(text)

        if not self.parse_trees[text] then
            local parse_trees, pts, words
            local w_net, u_net, v_net, matrix_networks, vector_networks, prediction_networks

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
                matrix_networks = make_matrix_networks(tree, self.U, self.V, self.A, self.WM)

                if self.max_repr_train > 0 then
                    predictor = self.repr_predictor or self.predictor
                else
                    predictor = nil
                end

                vector_networks, prediction_networks = make_vector_networks(
                    tree, matrix_networks, self.word_vecs, self.W,
                    self.non_linearity, predictor)

                nntools.update_table(self.matrix_networks, matrix_networks)
                nntools.update_table(self.vector_networks, vector_networks)
                nntools.update_table(self.prediction_networks, prediction_networks)
            end
        end

        return self.parse_trees[text], text
    end

    -- Return all parameters as an array of tensors.
    function MVRNN:parameters()
        local params, grad_params = {}, {}

        for word, word_vec in pairs(self.word_vecs) do
            params[#params + 1] = word_vec:get(2).bias
            grad_params[#grad_params + 1] = word_vec:get(2).gradBias
        end

        for word, U in pairs(self.U) do
            params[#params + 1] = U:get(2).bias
            grad_params[#grad_params + 1] = U:get(2).gradBias
        end

        for word, V in pairs(self.V) do
            params[#params + 1] = V:get(2).bias
            grad_params[#grad_params + 1] = V:get(2).gradBias
        end

        for word, A in pairs(self.A) do
            params[#params + 1] = A.bias
            grad_params[#grad_params + 1] = A.gradBias
        end

        params[#params + 1] = self.W:get(2).bias
        grad_params[#grad_params + 1] = self.W:get(2).gradBias

        params[#params + 1] = self.WM:get(2).bias
        grad_params[#grad_params + 1] = self.WM:get(2).gradBias

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

    function MVRNN:get_word_vec(word)
        return self.word_vecs[word]:get(2).bias
    end

    function MVRNN:get_U(word)
        return self.U[word]:get(2).bias
    end

    function MVRNN:get_V(word)
        return self.V[word]:get(2).bias
    end

    function MVRNN:get_A(word)
        return self.A[word].bias
    end

    function MVRNN:get_matrix(word)
        return self:get_U(word) * self:get_V(word) + torch.diag(self:get_A(word))
    end

    function MVRNN:clone(...)
        local no_clone = {
             'parse_trees', 'matrix_networks', 'vector_networks',
             'prediction_networks', 'forward_networks'}

        local tmp = {}
        for i, name in ipairs(no_clone) do
            tmp[name] = self[name]
            self[name] = {}
        end

        local clone = nn.Module.clone(self, ...)

        for name, obj in pairs(tmp) do
            self[name] = tmp[name]
        end

        return clone
    end

    function MVRNN:clone_all(...)
        return nn.Module.clone(self, ...)
    end
end

local function test_mvrnn_train(max_repr_train, do_manual, lambda)
    print("Testing MV-RNN...")
    n = 10
    r = 3
    non_linearity = nn.Tanh()

    mat_func = function (size) return torch.Tensor(size):normal(0, 0.3) end
    a_func = function (size) return torch.Tensor(size):normal(1, 0.05) end
    wv_func = function (size) return torch.Tensor(size):normal(0, 1.0) end
    -- wv_func = function (size) return (torch.rand(size)*0.02 - 0.01) end

    combine_mode = PROD
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
    local all_words = nntools.get_all_words(all_text)

    local mvrnn = MVRNN(
        n, r, all_words, non_linearity, predictor, repr_predictor, repr_criterion,
        combine_mode, mat_func, a_func, wv_func, false, max_repr_train)

    -- Very important that this happens before any text is fed to the network
    local w, dw = mvrnn:getParameters()

    print("Parameter norm: ", torch.norm(w))
    print("Gradient norm: ", torch.norm(dw))

    init_val = mvrnn:forward(text)
    init_vals = mvrnn:forward(batch)
    not_before = mvrnn:get_word_vec('not'):clone()
    not_mat_before = mvrnn:get_matrix('not'):clone()

    start = os.time()

    local max_steps = 100
    local n_steps = 0
    local grad_norm = 1.0
    while grad_norm > 0.01 and n_steps < max_steps do
        mvrnn:zeroGradParameters()

        -- Single instance mode
        mvrnn:accGradReprParameters(text, text_target)

        criterion:forward(mvrnn:forward(text), text_target)
        mvrnn:backward(text, criterion:backward(mvrnn.output, text_target))

        -- Batch mode
        mvrnn:accGradReprParameters(batch, batch_target)

        criterion:forward(mvrnn:forward(batch), batch_target)
        mvrnn:backward(batch, criterion:backward(mvrnn.output, batch_target))

        if do_manual then
            dw:add(lambda, w)
            w:add(-learning_rate, dw)
        else
            mvrnn:updateParameters(learning_rate)
        end

        grad_norm = torch.norm(dw)

        print("AFTER UPDATING: ")
        print("Parameter norm: ", torch.norm(w))
        print("Gradient norm: ", torch.norm(dw))
        print("n_steps: ", n_steps)

        n_steps = n_steps + 1
        learning_rate = learning_rate - 0.00001
    end

    final_val = mvrnn:forward(text)
    final_vals = mvrnn:forward(batch)
    not_after = mvrnn:get_word_vec('not'):clone()
    not_mat_after = mvrnn:get_matrix('not'):clone()

    finish = os.time()

    print("Training " .. n_steps .. " steps took " .. finish - start .. " seconds.")

    print("Initial: ")
    print(init_val)
    print(init_vals)
    print(not_before)
    print(not_mat_before)

    print("Final: ")
    print(final_val)
    print(final_vals)
    print(not_after)
    print(not_mat_after)
    print(mvrnn:get_U('not'))

    print "Passed :-)."
end

if arg[1] == 'test' or arg[1] == 'test_mvrnn' then
    test_mvrnn_train(30, false, 0.0)
    -- test_mvrnn_train(0, false, 0.0)
    test_mvrnn_train(30, true, 0.0)
    -- test_mvrnn_train(0, true, 0.0)
end

mvrnn = {
    MVRNN=MVRNN,
    make_mvrnn = function (
            n, r, words, non_linearity, predictor, repr_predictor,
            repr_criterion, combine_mode, mat_func, a_func, wv_func,
            allow_new_words, max_repr_train)

        return MVRNN(
            n, r, words, non_linearity, predictor, repr_predictor,
            repr_criterion, combine_mode, mat_func, a_func, wv_func,
            allow_new_words, max_repr_train), n
    end,

    get_all_words=nntools.get_all_words,

    CAT=CAT,
    MEAN=MEAN,
    SUM=SUM,
    PROD=PROD
}

return mvrnn
