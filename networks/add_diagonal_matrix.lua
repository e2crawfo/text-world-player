require 'torch'
require 'nn'

do
    local AddDiagonalMatrix, parent = torch.class("AddDiagonalMatrix", "nn.Module")

    function AddDiagonalMatrix:__init(n, init)
       parent.__init(self)

       self.size = n
       self.init = init
       if self.init then
           assert(init:dim() == 1 and init:size(1) == self.size)
           self.bias = init:clone()
       else
           self.bias = torch.zeros(self.size)
       end

       self.gradBias = torch.zeros(self.size)

       self:reset()
    end

    function AddDiagonalMatrix:reset(stdv)
       if self.init then
           self.bias = init:clone()
       else
           self.bias = torch.zeros(self.size)
       end

       self.gradBias = torch.zeros(self.size)
    end

    function AddDiagonalMatrix:updateOutput(input)
        self.output:resizeAs(input):copy(input)

        if input:dim() == 3 then
            for i = 1, self.size do
                self.output[{{}, {i}, {i}}]:add(self.bias[i])
            end
        elseif input:dim() == 2 then
            for i = 1, self.size do
                self.output[{{i}, {i}}]:add(self.bias[i])
            end
        else
            error("input has incorrect number of dimensions.")
        end

        return self.output
    end

    function AddDiagonalMatrix:updateGradInput(input, gradOutput)
       if self.gradInput then
          self.gradInput:resizeAs(gradOutput):copy(gradOutput) 
          return self.gradInput
       end
    end

    function AddDiagonalMatrix:accGradParameters(input, gradOutput, scale)
       scale = scale or 1
       if gradOutput:dim() == 3 then
           for i = 1, self.size do
               self.gradBias[{{i}}]:add(scale, torch.Tensor(gradOutput[{{}, {i}, {i}}]:sum(1)))
           end
       elseif input:dim() == 2 then
           for i = 1, self.size do
               self.gradBias[{{i}}]:add(scale, gradOutput[{{i}, {i}}])
           end
       else
           error("input has incorrect number of dimensions.")
       end
    end
end


function test_add_diagonal()
    local net = nn.Sequential()
    net:add(AddDiagonalMatrix(10))
    net:add(nn.Reshape(100))
    net:add(nn.Linear(100,1))

    local crit = nn.MSECriterion()

    local n_steps = 100
    local input = torch.zeros(10,10)
    local output = torch.Tensor(1.0)
    for i = 1, n_steps do
        net:zeroGradParameters()
        local err = crit:forward(net:forward(input), output)
        net:backward(input, crit:backward(net.output, output))
        net:updateParameters(0.01)
        print(err)
    end

    -- Batch mode
    net:reset()
    n_steps = 100
    input = torch.rand(80,10,10)
    output = torch.ones(80, 1):mul(2)
    for i = 1, n_steps do
        net:zeroGradParameters()
        err = crit:forward(net:forward(input), output)
        net:backward(input, crit:backward(net.output, output))
        net:updateParameters(0.01)
        print(err)
    end
end


if arg[1] == 'test' then
    test_add_diagonal()
end
