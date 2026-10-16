import math
import pytest
import torch
from rope_lab.positional import rotate, frequencies, causal_bias
from rope_lab.model import TinyDecoder, ModelConfig, METHODS
from rope_lab.tasks import batch, KEY0, BOS, LAG4, ASK

torch.set_num_threads(2)

def test_norms_composition_and_inverse():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(2,4,9,16,generator=g,dtype=torch.float64)
    p = torch.arange(9,dtype=torch.float64)*1234
    r = rotate(x,p)
    torch.testing.assert_close(r.norm(dim=-1),x.norm(dim=-1),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(rotate(r,-p),x,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(rotate(rotate(x,p),p+3),rotate(x,2*p+3),atol=1e-11,rtol=1e-11)

def test_relative_identity_and_shift():
    torch.manual_seed(8)
    q,k = torch.randn(2,1,32,dtype=torch.float64),torch.randn(2,1,32,dtype=torch.float64)
    m,n = torch.tensor([17.],dtype=torch.float64),torch.tensor([103.],dtype=torch.float64)
    left = (rotate(q,m)*rotate(k,n)).sum(-1)
    right = (q*rotate(k,n-m)).sum(-1)
    shifted = (rotate(q,m+3000)*rotate(k,n+3000)).sum(-1)
    torch.testing.assert_close(left,right,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(left,shifted,atol=1e-11,rtol=1e-11)

def test_dense_reference_catches_pairing_and_sign_errors():
    x = torch.tensor([[[1.,2.,3.,4.]]],dtype=torch.float64)
    p = torch.tensor([3.],dtype=torch.float64)
    theta = frequencies(4,dtype=torch.float64)
    blocks = [torch.tensor([[math.cos(3*t),-math.sin(3*t)],
                            [math.sin(3*t),math.cos(3*t)]],dtype=torch.float64) for t in theta]
    expected = x@torch.block_diag(*blocks).T
    torch.testing.assert_close(rotate(x,p),expected,atol=1e-12,rtol=1e-12)

def test_distance_does_not_force_decay():
    # Content can align exactly at an arbitrarily chosen distant offset.
    torch.manual_seed(9)
    q = torch.randn(1,1,64,dtype=torch.float64)
    d = torch.tensor([10000.],dtype=torch.float64)
    k = rotate(q,-d)
    far = (q*rotate(k,d)).sum()
    torch.testing.assert_close(far,q.square().sum(),atol=1e-10,rtol=1e-10)
    assert far > (q*k).sum()

def test_symmetric_damping_is_not_relative():
    q = torch.tensor([[[1.,0.]]],dtype=torch.float64)
    m,n = torch.tensor([2.]),torch.tensor([5.])
    gamma = 0.1
    def score(a,b):
        return ((math.exp(-gamma*a.item())*rotate(q,a)) *
                (math.exp(-gamma*b.item())*rotate(q,b))).sum()
    assert not torch.isclose(score(m,n),score(m+10,n+10))
    assert (n-m).item() == ((n+10)-(m+10)).item()

def test_interpolation_is_a_frequency_change():
    x = torch.randn(2,3,8,dtype=torch.float64)
    p = torch.arange(3,dtype=torch.float64)
    torch.testing.assert_close(rotate(x,p,scale=4),rotate(x,p/4))
    assert not torch.allclose(rotate(x,p),rotate(x,p,scale=4))

def test_rotation_gradient():
    x = torch.randn(1,3,4,dtype=torch.float64,requires_grad=True)
    p = torch.tensor([0.,3.,7.],dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda z: rotate(z,p), (x,))

@pytest.mark.parametrize("method",METHODS)
def test_cached_decoding_matches_full_prefix(method):
    torch.manual_seed(41)
    model = TinyDecoder(ModelConfig(method=method,dim=32,heads=4,layers=2)).eval()
    x = torch.randint(84,(2,13))
    with torch.no_grad():
        full = model(x)[0]
        cache,parts = None,[]
        for start,end in [(0,4),(4,5),(5,9),(9,13)]:
            y,cache = model(x[:,start:end],offset=start,past=cache)
            parts.append(y)
    torch.testing.assert_close(torch.cat(parts,dim=1),full,atol=2e-6,rtol=2e-5)

@pytest.mark.parametrize("method",METHODS)
def test_future_tokens_do_not_affect_past(method):
    torch.manual_seed(42)
    model = TinyDecoder(ModelConfig(method=method,dim=32,heads=4,layers=2)).eval()
    x = torch.randint(84,(2,11))
    y = x.clone()
    y[:,6:] = torch.randint(84,(2,5))
    with torch.no_grad():
        torch.testing.assert_close(model(x)[0][:,:6],model(y)[0][:,:6],atol=1e-6,rtol=1e-5)

def test_unmasked_attention_permutation_equivariance():
    torch.manual_seed(43)
    x = torch.randn(7,8,dtype=torch.float64)
    perm = torch.tensor([2,0,5,4,6,1,3])
    def attention(z):
        return (z@z.T/math.sqrt(8)).softmax(-1)@z
    torch.testing.assert_close(attention(x[perm]),attention(x)[perm])

def test_tasks_target_alignment_and_no_answer_leakage():
    g = torch.Generator().manual_seed(44)
    x,y = batch("lag4",32,16,g)
    assert (x[:,0]==BOS).all() and (x[:,-1]==LAG4).all()
    assert torch.equal(y,x[:,-5])
    x,y = batch("lookup",32,31,g)
    assert (x[:,-2]==ASK).all()
    for row,target in zip(x,y):
        keys,vals = row[1:-2:2],row[2:-2:2]
        assert len(keys.unique())==len(keys)
        assert torch.equal(vals[keys==row[-1]],target[None])

def test_shared_initialization_is_matched():
    states = []
    for method in METHODS:
        torch.manual_seed(11)
        states.append(TinyDecoder(ModelConfig(method=method)).state_dict())
    for key in states[0]:
        for state in states[1:]:
            torch.testing.assert_close(states[0][key],state[key],atol=0,rtol=0)

def test_mask_and_alibi_direction():
    p = torch.arange(4)
    b = causal_bias(p,p,4,alibi=True)
    assert torch.isneginf(b[0,0,1])
    assert b[0,3,0] < b[0,3,2] < b[0,3,3]

@pytest.mark.parametrize("dim",[0,3,-2])
def test_bad_rotary_dimensions(dim):
    with pytest.raises(ValueError):
        frequencies(dim)

def test_bad_position_and_scale():
    x = torch.zeros(2,3,8)
    with pytest.raises(ValueError):
        rotate(x,torch.arange(2))
    with pytest.raises(ValueError):
        rotate(x,torch.arange(3),scale=0)
