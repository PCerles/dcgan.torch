require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'cunn'
require 'image'

util = paths.dofile('util.lua')
nngraph.setDebug(true)
opt = {
   dataset = 'lsun',       -- imagenet / lsun / folder
   batchSize = 64,
   loadSize = 96,
   fineSize = 64,
   nz = 100,               -- #  of dim for Z
   ncond = 300, -- #  of dim for C
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 100,            -- #  of iter at starting learning rate
   lr = 0.0001,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'experiment1',
   noise = 'normal',       -- uniform / normal
   conditional = true,
   checkpoint = 100
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = 3
local nz = opt.nz
local ncond = opt.ncond
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0
local conditional = opt.conditional

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution
local JoinTable = nn.JoinTable

local nets = {}

function nets.generativeNet(ngf, noise_size, cond_size, conditional, out_size)
        -- input is Z + C, going into a convolution
        local noise = nn.Identity()()
        local cond = nn.Identity()()
    
        -- input is Z + C, going into a convolution
        local data = JoinTable(2,2)({noise, cond})
        local conv1 = SpatialFullConvolution(nz + ncond, ngf * 8, 4, 4)(data)
        local bn1 = SpatialBatchNormalization(ngf * 8)(conv1)
        local relu1 = nn.ReLU(true)(bn1)

        local conv2 = (SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))(relu1)
        local bn2 = SpatialBatchNormalization(ngf * 4)(conv2)
        local relu2 = nn.ReLU(true)(bn2)

        local conv3 = (SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))(relu2)
        local bn3 = SpatialBatchNormalization(ngf * 2)(conv3)
        local relu3 = nn.ReLU(true)(bn3)

        local conv4 = SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1)(relu3)
        local bn4 = SpatialBatchNormalization(ngf)(conv4)
        local relu4 = nn.ReLU(true)(bn4)

        local conv5 = SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1)(relu4)
        local output = nn.Tanh()(conv5)

        local netG = nn.gModule({noise,cond},{output})
        return netG
end

function nets.discriminativeNet(ndf)
    local data = nn.Identity()()
    local cond = nn.Identity()()
    
    -- Linear Layer applied to cond and data before network
    local reshData = nn.Reshape(nc*64*64)(data)
    local reshCond = nn.Reshape(ncond)(cond)
    local joinData = JoinTable(2,2)({reshData,reshCond})
    local firstLinear = nn.Linear(nc*64*64 + ncond,nc*64*64)(joinData)
    local inputData = nn.Reshape(nc,64,64)(firstLinear)
    
    -- input is (nc) x 64 x 64
    local conv1 = SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1)(inputData)
    local relu1 = nn.LeakyReLU(0.2, true)(conv1)
    
    -- state size: (ndf) x 32 x 32
    local conv2 = SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1)(relu1)
    local bn2 = SpatialBatchNormalization(ndf * 2)(conv2)
    local relu2 = nn.LeakyReLU(0.2, true)(bn2)
    
    -- state size: (ndf*2) x 16 x 16
    local conv3 = SpatialConvolution(ndf * 2 , ndf * 4, 4, 4, 2, 2, 1, 1)(relu2)
    local bn3 = SpatialBatchNormalization(ndf * 4)(conv3)
    local relu3 = nn.LeakyReLU(0.2, true)(bn3)
    
    -- state size: (ndf*4) x 8 x 8
    local conv4 = SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1)(relu3)
    local bn4 = SpatialBatchNormalization(ndf * 8)(conv4)
    local relu4 = nn.LeakyReLU(0.2, true)(bn4)
    
    -- state size: (ndf*8) x 4 x 4
    local conv5 = SpatialConvolution(ndf * 8, 1, 4, 4)(relu4)
    local sigmoid1 = nn.Sigmoid()(conv5)
    
    -- state size: 1 x 1 x 1
    local view1 = nn.View(1):setNumInputDims(3)(sigmoid1)
    local netD = nn.gModule({data,cond},{view1})
    return netD
end


if opt.checkpoint == 0 then
    netG = nets.generativeNet(ngf, noise_size, cond_size, conditional, out_size)
    netD = nets.discriminativeNet(ndf)
    netG:apply(weights_init)
    netD:apply(weights_init)
else
    netG = util.load('checkpoints_normal/' .. opt.name .. '_' .. opt.checkpoint .. '_net_G.t7', opt.gpu)
    netD = util.load('checkpoints_normal/' .. opt.name .. '_' .. opt.checkpoint .. '_net_D.t7', opt.gpu)
    print("loading nets from checkpoints")
end
local criterion = nn.BCECriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local cond = torch.Tensor(opt.batchSize, ncond, 1, 1)
local label = torch.Tensor(opt.batchSize)
local errD, errG, errN, nGrad
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  label = label:cuda(); cond = cond:cuda();
   netG = util.cudnn(netG);     netD = util.cudnn(netD);
   netD:cuda();   netG:cuda();  criterion:cuda()
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

noise_vis = noise:clone()

if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end

local fDxCond = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   local real, captions, scalarLabels, vocab_vecs = data:getBatch()
   data_tm:stop()
   
   
   input:copy(real)
   label:fill(real_label)
   cond:copy(captions)
   
   local output = netD:forward({input,cond}) -- run with image + conditional info
   local errD_real = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward({input,cond}, df_do)

   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end
   
   local fake = netG:forward({noise,cond})
   input:copy(fake)
   label:fill(fake_label)

   -- run discriminative net on generated image
   local output = netD:forward({input,cond})
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward({input,cond}, df_do)

   errD = errD_real + errD_fake
   return errD, gradParametersD
end

local fGxCond = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersG:zero()

   --[[ the three lines below were already executed in fDx, so save computation
   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward(noise)
   input:copy(fake) ]]--
   label:fill(real_label) -- fake labels are real for generator cost

   local output = netD.output -- netD:forward(input) was already executed in fDx, so save computation
   errG = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg = netD:updateGradInput({input,cond}, df_do)
   local df_di = df_dg[1]
   netG:backward({noise,cond},df_di)
   return errG,gradParametersG
end
local errDTable = {}
local errGTable = {}

saveTable = function(t, filename)
    local f = io.open(filename, 'w')
    for i=1,#t do
        f:write(t[i])
        f:write(',')
    end
    f:write('\n')
    f:close()
end


-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDxCond, parametersD, optimStateD)

      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGxCond, parametersG, optimStateG)

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
    table.insert(errDTable, errD and errD or -1)
    table.insert(errGTable, errG and errG or -1)
      end
   end
   paths.mkdir('checkpoints_normal')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   if epoch % 10 == 0 then
        real_epoch = opt.checkpoint + epoch
        util.save('checkpoints_normal/' .. opt.name .. '_' .. real_epoch .. '_net_G.t7', netG, opt.gpu)
        util.save('checkpoints_normal/' .. opt.name .. '_' .. real_epoch .. '_net_D.t7', netD, opt.gpu)
        saveTable(errDTable, 'plots_normal/errD_' .. opt.name .. '_' .. real_epoch .. '.txt')
        saveTable(errGTable, 'plots_normal/errG_' .. opt.name .. '_' .. real_epoch .. '.txt')
   end
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
