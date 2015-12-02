require 'nn'
require 'cunn'
require 'torch'
require 'dpnn'
require 'parse'
require 'text_to_vector'


local function filter(t, f)
    local out = {}

    for k, v in pairs(t) do
        if f(k, v) then out[k] = v end
    end

    return out
end


local function update_table(base, new, overwrite)
    overwrite = overwrite or true
    for k, v in pairs(new) do
        if not base[k] or overwrite then
            base[k] = v
        end
    end
end

function concat_array(t1, t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end

local function clone_net(net)
    return net:clone('weight', 'bias')
end


-- Possible values for ``combine_mode``
local CAT = 'cat'
local SUM = 'sum'
local MEAN = 'mean'


local function make_matrix_networks (tree, U, V, WM)
    local matrix_networks = {}

    function f (tree)
        local concat, network, left_network, right_network

        if tree.terminal then
            concat = nn.ConcatTable():add(clone_net(U[tree.word])):add(clone_net(V[tree.word]))
            network = nn.Sequential():add(concat):add(nn.MM())
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


local function make_vector_networks (tree, matrix_networks, word_vecs, W, nl_class, predictor)

    local vector_networks = {}
    local prediction_networks = {}

    function f(tree)
        if tree.terminal then
            return clone_net(word_vecs[tree.word])
        end

        local left_vector, left_matrix, left_stream
        local right_vector, right_matrix, right_stream
        local concat, network

        left_vector = f(tree.left)
        left_matrix = clone_net(matrix_networks[tree.left])

        right_vector = f(tree.right)
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
        network:add(nl_class())

        vector_networks[tree] = nn.Sequential():add(network):add(nn.View(W:get(1).size[1]))

        prediction_networks[tree] = (
            nn.Sequential()
            :add(vector_networks[tree])
            :add(clone_net(predictor)))

        return network
    end

    f(tree)
    return vector_networks, prediction_networks
end

do
    local MVRNN, parent = torch.class('MVRNN', 'Text2Vector')

    function MVRNN:__init(n, r, nl_class, combine_mode, rv_func, wv_func)
        -- combine_mode is one of MEAN, SUM, CAT
        -- wv_func is an optional function for setting initial word vectors
        -- if wv_func not supplied, fall back on rv_func to initialize the word vectors

        text_to_vector.Text2Vector.__init(self, n, false)

        self.random_vec_func = rv_func
        self.wv_func = wv_func or rv_func

        self.nl_class = nl_class or nn.Tanh
        self.combine_mode = combine_mode or MEAN

        W_shape = torch.LongStorage{n, 2*n}
        W = nn.Sequential():add(nn.Constant(torch.zeros(W_shape))):add(nn.Add(W_shape))
        W:get(1).size = W_shape
        W:get(2).bias = self.random_vec_func(W_shape)

        self.W = W

        WM_shape = torch.LongStorage{n, 2*n}
        WM = nn.Sequential():add(nn.Constant(torch.zeros(WM_shape))):add(nn.Add(WM_shape))
        WM:get(1).size = WM_shape
        WM:get(2).bias = self.random_vec_func(WM_shape)

        self.WM = WM

        networks = {}

        -- A mapping: text fragment -> (sentence -> parse tree)
        self.parse_trees = {}

        self.U = {}
        self.V = {}
        self.word_vecs = {}

        self.u_shape = torch.LongStorage{n, r}
        self.v_shape = torch.LongStorage{r, n}
        self.w_shape = torch.LongStorage{n, 1}

        self.matrix_networks = {}
        self.vector_networks = {}
        self.prediction_networks = {}
    end

    -- Called from text_to_vector.Text2Vector.updateOutput
    function MVRNN:fillVector(text, vector)
        self:parse(text)

        local vectors = nil
        for s, tree in pairs(self.parse_trees[text]) do
            result = self.vector_networks[tree]:forward(torch.Tensor())

            row = result:reshape(1, result:nElement())
            if not vectors then
                vectors = row
            else
                vectors = torch.cat(vectors, row, 1)
            end
        end

        if self.combine_mode == MEAN then
            vector:copy(vectors:mean(1):squeeze())
        elseif self.combine_mode == SUM then
            vector:copy(vectors:sum(1):squeeze())
        elseif self.combine_mode == CAT then
            vector:copy(vectors:view(vectors:nElement()))
        else
            error("Invalid value for combine_mode: " .. self.combine_mode)
        end
    end


    function MVRNN:setup_training(predictor, criterion)
        self.predictor = predictor
        self.criterion = criterion or nn.MSECriterion()
    end


    function MVRNN:train(text, target, learning_rate)
        -- Update parameters of the MV-RNN on an instance where ``input``
        -- is a sentence or paragraph, and ``value`` is the value to predict
        -- based on the constituents. The type of ``value`` is dependent
        -- on what was passed into ``setup_training``.

        local typename, size

        typename = torch.typename(text)
        if type(text) == 'table' or typename and text:dim() > 1 then
            -- Batch mode
            size = type(text) == 'table' and #text or text:size(1)
            assert(
                (type(target) == 'table' and #text == #target)
                or (torch.typename(target) and #text == targ:size(1)))

        elseif typename and text:dim() == 1 or type(text) == 'string' then
            size = 1
            text = {text}
            target = {target}
        else
            error("Invalid input type. Expected vector, matrix, string or array of strings.")
        end

        local input, trgt, constituent_subtrees, nets_to_train

        nets_to_train = {}

        for i=1,size do
            txt_i = self:preprocessText(text[i])
            self:parse(txt_i)

            trgt = torch.Tensor{target[i]}

            for s, tree in pairs(self.parse_trees[txt_i]) do
                constituent_subtrees = filter(
                    tree:get_subtrees(),
                    function (t, v) return not t.terminal end)

                for t, _ in pairs(constituent_subtrees) do
                    nets_to_train[#nets_to_train+1] = {self.prediction_networks[t], trgt}
                end
            end
        end

        for i, net in ipairs(nets_to_train) do
            net[1]:zeroGradParameters()
        end

        input = torch.Tensor()
        for i, net in ipairs(nets_to_train) do
            self.criterion:forward(net[1]:forward(input), net[2])
            net[1]:backward(input, self.criterion:backward(net[1].output, net[2]))
        end

        for i, net in ipairs(nets_to_train) do
            net[1]:updateParameters(learning_rate/size)
        end
    end


    function MVRNN:parse(input)
        if torch.typename(input) == "torch.ByteTensor" then
            input = input:storage():string()
        end

        assert(type(input) == 'string', 'Input must be string-like.')

        if not self.parse_trees[input] then
            print("Parsing: " .. input)

            local parse_trees, pts, words, u_shape, v_shape, w_shape
            local w_net, u_net, v_net, matrix_networks, vector_networks, prediction_networks

            parse_trees = parse.parse_sentences({input})

            -- Use the first parse.
            pts = {}
            for s, trees in pairs(parse_trees) do
                pts[s] = trees[1]
            end
            parse_trees = pts
            self.parse_trees[input] = parse_trees

            words = {}
            for s, tree in pairs(parse_trees) do
                for w, v in pairs(tree:get_words()) do
                    words[w] = v
                end
            end

            u_shape, v_shape, w_shape = self.u_shape, self.v_shape, self.w_shape

            for word, _ in pairs(words) do
                if not self.word_vecs[word] then
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
                    u_net:get(2).bias = self.random_vec_func(u_shape)
                    self.U[word] = u_net

                    v_net = (
                        nn.Sequential()
                        :add(nn.Constant(torch.zeros(v_shape)))
                        :add(nn.Add(v_shape)))

                    v_net:get(1).size = v_shape
                    v_net:get(2).bias = self.random_vec_func(v_shape)
                    self.V[word] = v_net
                end
            end

            for s, tree in pairs(parse_trees) do
                matrix_networks = make_matrix_networks(tree, self.U, self.V, self.WM)

                vector_networks, prediction_networks = make_vector_networks(
                    tree, matrix_networks, self.word_vecs, self.W,
                    self.nl_class, self.predictor)

                update_table(self.matrix_networks, matrix_networks)
                update_table(self.vector_networks, vector_networks)
                update_table(self.prediction_networks, prediction_networks)
            end
        end
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

    function MVRNN:get_matrix(word)
        return self:get_U(word) * self:get_V(word)
    end

    function MVRNN:predict(text)
        local typename, size, txt_i, result, output

        typename = torch.typename(text)
        if type(text) == 'table' or typename and text:dim() > 1 then
            -- Batch mode
            size = type(text) == 'table' and #text or text:size(1)
        elseif typename and text:dim() == 1 or type(text) == 'string' then
            size = 1
            text = {text}
        else
            error("Invalid input type. Expected vector, matrix, string or array of strings.")
        end

        output = {}
        for i=1,size do
            txt_i = self:preprocessText(text[i])
            self:parse(txt_i)

            for s, tree in pairs(self.parse_trees[txt_i]) do
                result = self.prediction_networks[tree]:forward(torch.Tensor())
                output[#output + 1] = result:storage()[1]
            end
        end

        return torch.Tensor(output)
    end
end

local function test_mvrnn()
    print("Testing MV-RNN...")
    n = 20
    r = 3
    nl_class = nn.Tanh
    random_vec_func = function (size) return torch.Tensor(size):normal(0, 1.0) end
    combine_mode = MEAN

    learning_rate = 0.01

    local mvrnn = MVRNN(n, r, nl_class, combine_mode, random_vec_func)
    mvrnn:setup_training(nn.Linear(n, 1))

    text = "This is the end. The dog ran to the end."
    batch = {"The end of the dog.", "We won a game.", "The end of the world is coming soon.", "The end of the world is not coming."}
    n_steps = 250

    initial_vals = mvrnn:predict(batch)
    init_val = mvrnn:predict(text)

    start = os.time()
    for i=1,n_steps do
        mvrnn:train(text, 0.0, learning_rate)
        print(mvrnn:predict(text))

        -- Batch mode
        mvrnn:train(batch, {1.0, 2.0, 3.0, 4.0}, learning_rate)
        print(mvrnn:predict(batch))
    end

    final_vals = mvrnn:predict(batch)
    final_val = mvrnn:predict(text)

    finish = os.time()

    print("Training " .. n_steps .. " steps took " .. finish - start .. " seconds.")

    print("Initial: ")
    print(initial_vals)
    print(init_val)

    print("Final: ")
    print(final_vals)
    print(final_val)
end

if false then
    test_mvrnn()
end

mvrnn = {
    MVRNN=MVRNN,
    make_mvrnn = function (n, r, nl_class, combine_mode, rv_func, wv_func)
        return MVRNN(n, r, nl_class, combine_mode, rv_func, wv_func), n
    end,

    BINARY=BINARY,
    MULTICLASS=MULTICLASS,
    CTS=CTS,

    CAT=CAT,
    MEAN=MEAN,
    SUM=SUM
}

return mvrnn
