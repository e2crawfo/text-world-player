require 'nn'
require 'torch'
require 'dpnn'
require 'nntools'
require 'mvrnn'

local clone_net = nntools.clone_net

local function trim_string(s)
  return s and s:match'^%s*(.*%S)' or nil
end

local function text2sentences(text)
    iterator = string.gmatch(text, '[^.]+')

    local function f()
        return trim_string(iterator())
    end

    return f
end

do
    local RandomVectorPerSentence = torch.class('RandomVectorPerSentence', 'MVRNN')

    function RandomVectorPerSentence:__init(text, combine_mode, rv_func, dimension, modify_vecs, allow_new_sentences)
        self.dimension = dimension
        self.combine_mode = combine_mode
        self.rv_func = rv_func
        self.allow_new_sentences = allow_new_sentences
        self.modify_vecs = modify_vecs

        self.sentence_networks = {}
        self.forward_networks = {}
        self.output = torch.Tensor()

        self:add_text(text)
    end

    function RandomVectorPerSentence:add_text(text)
        for sentence in text2sentences(text) do
            self:add_sentence(sentence)
        end
    end

    function RandomVectorPerSentence:add_sentence(sentence)
        if not self.sentence_networks[sentence] then
            local shape = torch.LongStorage{self.dimension, 1}
            local s_net = (
                nn.Sequential()
                :add(nn.Constant(torch.zeros(shape)))
                :add(nn.Add(shape)))

            s_net:get(1).size = shape
            s_net:get(2).bias = self.rv_func(shape)

            self.sentence_networks[sentence] = s_net
        end
    end

    function RandomVectorPerSentence:updateOutput(text, vector)
        -- Split text up into sentences.

        local batch_mode, size, pts, txt_i, concat, forward_net, result
        batch_mode, size, text = self:handleBatch(text)

        self.output = nil

        for i=1,size do
            txt_i = self:preprocessText(text[i])
            forward_net = self.forward_networks[txt_i]

            if not forward_net then
                concat = nn.ConcatTable()
                for sentence in text2sentences(txt_i) do
                    concat:add(self.sentence_networks[sentence])
                end

                forward_net = nn.Sequential():add(concat)

                if self.combine_mode == mvrnn.MEAN then
                    print("USING MEAN")
                    forward_net:add(nn.JoinTable(2)):add(nn.Mean(2))
                elseif self.combine_mode == mvrnn.SUM then
                    forward_net:add(nn.JoinTable(2)):add(nn.Sum(2))
                elseif self.combine_mode == mvrnn.CAT then
                    forward_net:add(nn.JoinTable(1))
                elseif self.combine_mode == mvrnn.PROD then
                    print("USING PROD")
                    forward_net:add(nn.CMulTable()):add(nn.Reshape(self.dimension))
                else
                    error("Invalid value for combine_mode: " .. self.combine_mode)
                end

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

    function RandomVectorPerSentence:accGradReprParameters(text, target, mask)
        error("Representation of RandomVectorPerSentence cannot be trained.")
    end

    function RandomVectorPerSentence:accGradParameters(text, gradOutput, scale)
        local batch_mode, size, txt_i

        if self.modify_vecs then
            batch_mode, size, text, gradOutput = self:handleBatch(text, gradOutput)

            -- Train the entire network
            for i=1,size do
                txt_i = self:preprocessText(text[i])
                forward_net = self.forward_networks[txt_i]
                forward_net:backward(torch.Tensor(), gradOutput[i])
            end
        end
    end

    -- Return all parameters as an array of tensors.
    function RandomVectorPerSentence:parameters()
        local params, grad_params = {}, {}

        for sentence, sent_vec in pairs(self.sentence_networks) do
            params[#params + 1] = sent_vec:get(2).bias
            grad_params[#grad_params + 1] = sent_vec:get(2).gradBias
        end

        return params, grad_params
    end

    function RandomVectorPerSentence:get_sentence_vec(sentence)
        local lc = sentence:sub(#sentence)
        if lc == '.' or lc == '?' or lc == '!' or lc == ',' then
            sentence = sentence:sub(1, #sentence-1)
        end

        return self.sentence_networks[trim_string(sentence)]:get(2).bias
    end

    function RandomVectorPerSentence:clone(...)
        return nn.Module.clone(self, ...)
    end
end

local function make_rvps(text, combine_mode, rv_func, dimension, modify_vecs, allow_new_sentences)
    local rvps = RandomVectorPerSentence(text, combine_mode, rv_func, dimension, modify_vecs, allow_new_sentences)
    return rvps, rvps.dimension
end


local function test_rvps(combine_mode, modify_vecs, do_manual, lambda)
    print("Testing RVPS..." .. string.rep("+", 80))
    n = 10
    lambda = lambda or 0.0

    -- rv_func = function (size) return torch.Tensor(size):bernoulli() end
    rv_func = function (size) return torch.Tensor(size):normal(0, 1.0) end

    predictor = nn.Linear(n, 1)

    learning_rate = 0.03

    text = "The end is near. They won the game. Its almost over."
    text_target = torch.Tensor{1.0}

    batch = {
        "The end of the dog. The end of the game. Its almost over.",
        "The end is near. Its not far now. The dog's paw is injured.",
        "The end is near. Its close. The dog's paw is injured."}

    batch_target = torch.Tensor{{1.0, 2.0, 4.0}}:t()

    criterion = nn.MSECriterion()

    -- Get all words.
    local all_text = table.concat(batch, ' ') .. " " .. text

    local rvps = (
        nn.Sequential()
        :add(RandomVectorPerSentence(all_text, combine_mode, rv_func, n, modify_vecs, false))
        :add(predictor))

    -- Very important that this happens before any text is fed to the network
    local w, dw = rvps:getParameters()

    print("Parameter norm: ", torch.norm(w))
    print("Gradient norm: ", torch.norm(dw))

    init_val = rvps:forward(text):clone()
    init_vals = rvps:forward(batch):clone()
    not_before = rvps:get(1):get_sentence_vec('The end of the game.'):clone()

    start = os.time()

    local max_steps = 100
    local n_steps = 0
    local grad_norm = 1.0
    while grad_norm > 0.01 and n_steps < max_steps do
        rvps:zeroGradParameters()

        -- Single instance mode
        criterion:forward(rvps:forward(text), text_target)
        rvps:backward(text, criterion:backward(rvps.output, text_target))

        -- Batch mode
        criterion:forward(rvps:forward(batch), batch_target)
        rvps:backward(batch, criterion:backward(rvps.output, batch_target))

        if do_manual then
            dw:add(lambda, w)
            w:add(-learning_rate, dw)
        else
            rvps:updateParameters(learning_rate)
        end

        grad_norm = torch.norm(dw)

        -- print("AFTER UPDATING: ")
        -- print("Parameter norm: ", torch.norm(w))
        -- print("Gradient norm: ", torch.norm(dw))
        -- print("n_steps: ", n_steps)

        n_steps = n_steps + 1
        learning_rate = learning_rate - 0.00001
    end

    final_val = rvps:forward(text):clone()
    final_vals = rvps:forward(batch):clone()
    not_after = rvps:get(1):get_sentence_vec('The end of the game.'):clone()

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

    return n_steps
end

if arg[1] == 'test' or arg[1] == 'test_rvps' then
    print ("Using MEAN: " .. string.rep("*", 40))
    test_rvps(mvrnn.MEAN, true, false)
    test_rvps(mvrnn.MEAN, true, true)

    print ("Using PROD: " .. string.rep("*", 40))
    test_rvps(mvrnn.PROD, true, false)
    test_rvps(mvrnn.PROD, true, true)

    print ("Using MEAN: " .. string.rep("*", 40))
    test_rvps(mvrnn.MEAN, false, false)
    test_rvps(mvrnn.MEAN, false, true)

    print ("Using PROD: " .. string.rep("*", 40))
    test_rvps(mvrnn.PROD, false, false)
    test_rvps(mvrnn.PROD, false, true)
end


sentence_vectors = {
    RandomVectorPerSentence=RandomVectorPerSentence,
    make_rvps=make_rvps,

    CAT=CAT,
    MEAN=MEAN,
    SUM=SUM,
    PROD=PROD
}

return sentence_vectors
