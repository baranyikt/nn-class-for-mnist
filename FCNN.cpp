// FCNN : Fully Connected Neural Network
// Fully connected neural network model with backpropagation (especially for MNIST data)
//
// Written by: Baranyi Karoly 
// 2017-11-23
// 
// 2021-11-18 added very simple implementation of weight normalization described in [1] Tim Salimans, Diederik P. Kingma: Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks
// 2021-11-22 nonlinearity detached from fully connected layer

#define USE_WEIGHT_NORMALIZATION

#include <vector>
#include <list>
#include <memory>
#include <cassert>
#include <valarray>
#include <functional>
#include <fstream>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <random>
#include <string>
#include <map>
#include <locale>         
#include <algorithm>	
#include <sstream>
#include <vector>
#include <memory>
#include <valarray>

// types
using std::vector;
using std::unique_ptr;
using std::valarray;
using std::string;

using std::cout;
using std::cerr;
using std::endl;

using sca_t = float;												// scalar type for arithmetic operators, float/double/long double
using vec_t = valarray<sca_t>;										// (math) vector and matrix types based on that
using mtx_t = valarray<sca_t>;

using acfunc_t = sca_t(__cdecl *)(const sca_t);						// funcptr type for activation function and its derivative
using erfunc_t = sca_t(__cdecl *)(const vec_t&, const vec_t&);		// ... for error function
using erfuncd_t = vec_t(__cdecl *)(const vec_t&, const vec_t&);		// ... for error function's derivative

// consts
constexpr unsigned INPUT_LAYER_SIZE = 28 * 28;						// first layer input dimension's fixed
constexpr unsigned TERMINAL_LAYER_SIZE = 10;						// as the output of the last layer
constexpr uint32_t TRAINIMAGES_MAGIC = 0x00000803;					// MNIST-specific magic numbers
constexpr uint32_t TRAINLABELS_MAGIC = 0x00000801;
constexpr uint32_t TESTIMAGES_MAGIC = 0x00000803;
constexpr uint32_t TESTLABELS_MAGIC = 0x00000801;

const string TRAINIMAGES_DEFAULT = "train-images.idx3-ubyte";		// default values for parameters below, see there
const string TRAINLABELS_DEFAULT = "train-labels.idx1-ubyte";
const string TESTIMAGES_DEFAULT = "t10k-images.idx3-ubyte";
const string TESTLABELS_DEFAULT = "t10k-labels.idx1-ubyte";
constexpr unsigned PASSES_DEFAULT = 1;
constexpr unsigned TRUNCDATASIZE_DEFAULT = ULONG_MAX;
constexpr unsigned BATCHSIZE_DEFAULT = 1;
constexpr auto LEARNINGRATE_DEFAULT = static_cast<sca_t>(0.1f);
constexpr unsigned NORMALIZE_DEFAULT = 1;
constexpr unsigned SHOWCURVE_DEFAULT = 0;
constexpr unsigned SHOWCURVETRUNC_DEFAULT = 0;

constexpr sca_t DEFAULT_MTXMEAN = 0;
constexpr sca_t DEFAULT_MTXDEV = static_cast<sca_t>(0.01f);

const string LOADSAVEMATRIX_DEFAULT = "";
const string SAVEMATRIX_DEFAULT = "";
const string LOADMATRIX_DEFAULT = "";
const string MATRIXCURVE_DEFAULT = "";

// global vars

bool global_truth_conv10_one_hot_conversion = true;					// need to set to true for MNIST dataset, but often false for testing/debugging sets
string global_trainimages_fn = TRAINIMAGES_DEFAULT;					// full paths of datafiles
string global_trainlabels_fn = TRAINLABELS_DEFAULT;
string global_testimages_fn = TESTIMAGES_DEFAULT;
string global_testlabels_fn = TESTLABELS_DEFAULT;
unsigned global_passes = PASSES_DEFAULT;							// number training passes on the whole training set
unsigned global_truncdatasize = TRUNCDATASIZE_DEFAULT;				// != ULONG_MAX if training set is to be truncated (for speed reasons)
unsigned global_batchsize = BATCHSIZE_DEFAULT;						// 
sca_t global_learningrate = LEARNINGRATE_DEFAULT;					// LR
unsigned global_normalize_enabled = NORMALIZE_DEFAULT;				// transforms training and test data to have 0 mean and 1 deviance
unsigned global_showcurve_enabled = SHOWCURVE_DEFAULT;				// if true (1), displays learning rate statistics after every batch
unsigned global_showcurve_trunc = SHOWCURVETRUNC_DEFAULT;			// != 0 if learning rate stats should be based on a truncated dataset (for speed reasons)
sca_t global_mtxmean = DEFAULT_MTXMEAN;								// desired expected value in random matrix generator
sca_t global_mtxdev = DEFAULT_MTXDEV;								// ... deviance

string global_savematrix_fn = SAVEMATRIX_DEFAULT;					// if not empty, save the resulting matrix at the very end to this path
string global_loadmatrix_fn = LOADMATRIX_DEFAULT;					// saved matrices from previous runs can be loaded at startup instead of using the default
string global_loadsavematrix_fn = LOADSAVEMATRIX_DEFAULT;			// common name for the previous two (if not empty, overrides those)

string global_matrixcurve_fn = MATRIXCURVE_DEFAULT;					// saves matrix (log-style) here after every batch (if not empty)

std::map<string, std::pair<acfunc_t, acfunc_t>> global_actFns;		// activation functions to be chosen from (and their derivatives)
std::map<string, std::pair<erfunc_t, erfuncd_t>> global_errFns;		// error functions

std::map<string, string*> global_strOpts;							// command line argument name --> global variable mappings
std::map<string, unsigned*> global_intOpts;
std::map<string, sca_t*> global_floatOpts;

class NeuralNetwork;

// abstract class representing one layer in neural network (with unspecified connectedness)
class NNLayer {
protected:
	NeuralNetwork* _mothernetwork = nullptr;
public:
	NNLayer() = default;
	virtual ~NNLayer() = default;
	virtual void LinkLayer(NeuralNetwork* mothernetwork, unsigned nextLayerSize) = 0;
	virtual unsigned GetInputSize() const = 0;
	virtual vec_t FwdProp(vec_t& input) = 0;
	virtual vec_t BackProp(vec_t& Wdelta) = 0;
	virtual void UpdateWeightMatrix() = 0;
	virtual void InitBatch() = 0;
	virtual bool IsValid() const = 0;
	virtual void SetLearningRate(const sca_t& learningrate) = 0;
};

// class for applying non-linearity after fully connected layer or any other kind of layer
class NNNonLinearity : public NNLayer {
private:
	unsigned	_outputsize = 0;
	unsigned	_inputsize = 0;
	vec_t		_input;
	vec_t		_output;
	acfunc_t	_activFn;
	acfunc_t	_activFnDerived;
public:
	NNNonLinearity(acfunc_t activationFunction, acfunc_t derivedFunction) :
		_activFn(activationFunction), _activFnDerived(derivedFunction) {}
	virtual void LinkLayer(NeuralNetwork* mothernetwork, unsigned nextLayerSize) override;
	virtual vec_t FwdProp(vec_t& input) override;
	virtual vec_t BackProp(vec_t& Wdelta) override;
	virtual void UpdateWeightMatrix() override {}
	virtual void InitBatch() override {}
	virtual unsigned GetInputSize() const override { return _inputsize; }
	virtual void SetLearningRate(const sca_t& learningrate) override {}
	virtual bool IsValid() const override;
};

// class representing a fully connected layer in the network -- without nonlinearity, only the affine transformation
class NNFullyConnected : public NNLayer {
private:
	unsigned	_outputsize = 0;
	unsigned	_inputsize = 0;
#ifdef USE_WEIGHT_NORMALIZATION
	mtx_t		_weightupdate_v;
	vec_t		_weightupdate_g;
#else
	mtx_t		_weightupdate;
#endif
	unsigned	_weightmatrix_rows = 0;
	unsigned	_weightmatrix_cols = 0;
private:
	unsigned	_batchcnt = 0;
	vec_t		_input;
	vec_t		_output;
	sca_t		_learningrate = LEARNINGRATE_DEFAULT;
#ifdef USE_WEIGHT_NORMALIZATION
	mtx_t		_weightmatrix_v;		// v,g, in [1] eq (2)
	vec_t		_weightmatrix_g;
#else
	mtx_t		_weightmatrix;					
#endif
	vec_t		_bias;
	vec_t		_biasupdate;
	void InitMatrix(unsigned n, unsigned m);
public: 
	NNFullyConnected(unsigned inputsize) :
		_inputsize(inputsize) {};
	virtual void LinkLayer(NeuralNetwork* mothernetwork, unsigned nextLayerSize) override;
	virtual vec_t FwdProp(vec_t& input) override;
	virtual vec_t BackProp(vec_t& Wdelta) override;
	virtual void UpdateWeightMatrix() override;
	virtual void InitBatch() override;
	virtual unsigned GetInputSize() const override { return _inputsize; }
	virtual void SetLearningRate(const sca_t& learningrate) override { _learningrate = learningrate; };
	virtual bool IsValid() const override;
	friend bool LoadMatrix(const string& fn, NeuralNetwork& nn);
	friend bool SaveMatrix(const string& fn, NeuralNetwork& nn, bool because_mtxcurve);
private:
#ifdef USE_WEIGHT_NORMALIZATION
	mtx_t CalcWeightMatrixFromVandG() const;
	void CalcGandVupdateFromWupdate(const mtx_t& wu, mtx_t& vu, vec_t& gu) const;
#endif
};

// class for pseudo layer at the end of network
class NNTerminal : public NNLayer {
private:
	vec_t		_truth;
	erfunc_t	_errorFn;
	erfuncd_t	_errorFnDerived;
	vec_t		_input;
	unsigned	_inputsize = 0;
public:
	NNTerminal(unsigned inputsize, erfunc_t errorFn, erfuncd_t errorFnDerived) :
		_inputsize(inputsize), _errorFn(errorFn), _errorFnDerived(errorFnDerived) {}
	virtual void LinkLayer(NeuralNetwork* mothernetwork, unsigned nextLayerSize) override;
	virtual unsigned GetInputSize() const override { return _inputsize; }
	virtual vec_t FwdProp(vec_t& input) override;
	virtual vec_t BackProp(vec_t& Wdelta) override;
	virtual void UpdateWeightMatrix() override {};
	virtual void setTruth(vec_t& truth) { _truth = truth; };
	virtual void InitBatch() override {};
	virtual bool IsValid() const override;
	virtual void SetLearningRate(const sca_t& learningrate) override {};
	sca_t GetResult() const;
	unsigned IsAccurate() const;
};

class NeuralNetwork {
private:
	vector<unique_ptr<NNLayer>> _layers;
public:
	NeuralNetwork() = default;
	~NeuralNetwork() = default;
	NNLayer* GetLastLayer() { return (_layers.size() == 0 ? nullptr : _layers.back().get()); };
	void FwdPropThrough(const vec_t& initial_input);
	void BackPropThrough();
	void UpdateWeightsThrough();
	bool IsValid() const;
	void SetLearningRate(const sca_t& learningrate);
	void InitBatchAll();
	void LinkLayerAll();
	friend NeuralNetwork& operator<< (NeuralNetwork&, unique_ptr<NNLayer>&&);
	friend bool LoadMatrix(const string& fn, NeuralNetwork& nn);
	friend bool SaveMatrix(const string& fn, NeuralNetwork& nn, bool because_mtxcurve);
};

class DBManager {
public:
	vector<vec_t>	trainimages;
	vector<sca_t>	trainlabels;
	vector<vec_t>	testimages;
	vector<sca_t>	testlabels;
	bool ReadDB(const string & fn_images, const string & fn_labels, uint32_t magic_images, uint32_t magic_labels, vector<vec_t>& images, vector<sca_t>& labels);
	void CalcStatParams(vector<vec_t>& data, sca_t& mean, sca_t& dev);
	void NormalizeDB(vector<vec_t>& data, const sca_t& mean, const sca_t& dev);
	void FeedDatabase(const vector<vec_t>& database, const vector<sca_t>& truthdb, NeuralNetwork & neuralnet, const unsigned & batchSize, const sca_t & learningRate);
	void TestDatabase(const vector<vec_t>& database, const vector<sca_t>& truthdb, NeuralNetwork & neuralnet, unsigned showcurve_trunc = 0, bool silent = false);
	void PrintMatrixToConsole(const mtx_t & mtx, unsigned rows, unsigned cols);
	void DrawOneLetterToConsole(bool testOrTRAIN, int nr, int threshold = 128);
	static uint32_t ReadUint32MSBF(unsigned char*& bufseek);
};

// helper functions for command-line argument checks

inline string StrToUpper(const string& s) {
	std::locale loc;												// <--- set your console locale here if needed!
	string value = s;
	for (unsigned i = 0; i < s.length(); ++i) {
		value[i] = std::toupper(s[i], loc);
	}
	return value;
}

inline bool IsInteger(const string& s) {
	std::istringstream iss(s);
	int i;
	iss >> std::noskipws >> i;
	return iss.eof() && !iss.fail();
}

inline bool IsFloat(const string& s) {
	std::istringstream iss(s);
	float f;
	iss >> std::noskipws >> f;
	return iss.eof() && !iss.fail();
}

// activation functions / error functions and their helpers

inline vec_t stable_softmax(const vec_t& ofWhat) {
	vec_t shifted = ofWhat - ofWhat.max();
	for (unsigned i = 0; i < shifted.size(); ++i) { shifted[i] = exp(shifted[i]); }
	shifted /= shifted.sum();
	return shifted;
}

inline sca_t cross_entropy(const vec_t& T, const vec_t& P) {
	assert(T.size() == P.size());
	sca_t value = 0;
	for (unsigned i = 0; i < T.size(); ++i) {
		value -= T[i] * log(P[i]);
	}
	return value;
}

// tanh activation function and its derivative

sca_t __cdecl ac_tanh(const sca_t x) {
	return std::tanh(x);
}

sca_t __cdecl ac_tanh_d(const sca_t x) {
	return 1.0f - pow(ac_tanh(x), 2);
}

// rectified linear

sca_t __cdecl relu(const sca_t x) {
	return (x > 0 ? x : 0);
}

sca_t __cdecl relu_d(const sca_t x) {
	return static_cast<sca_t>((x > 0 ? 1 : 0));
}

// sigmoid

sca_t __cdecl stable_sigmoid(const sca_t x) {
	if (x >= 0) {
		sca_t z = std::exp(-x);
		return static_cast<sca_t>(1.0 / (1.0 + z));
	}
	else {
		sca_t z = std::exp(x);
		return static_cast<sca_t>(z / (1.0 + z));
	}
}

sca_t __cdecl sigmoid_d(const sca_t x) { 
	return static_cast<sca_t>(stable_sigmoid(x) * (1.0 - stable_sigmoid(x))); 
}

// smooth relu

sca_t srelu_factor = 1;

sca_t __cdecl srelu(const sca_t x) {
	return std::log(1 + std::exp(x * srelu_factor)) / srelu_factor;
}

sca_t __cdecl srelu_d(const sca_t x) {
	return stable_sigmoid(x);
}

// identity as activation function

sca_t __cdecl identity(const sca_t x) { return x; }
sca_t __cdecl identity_d(const sca_t x) { return 1; }

// MSE error function

sca_t __cdecl squared_err(const vec_t& truth, const vec_t& input) {
	assert(truth.size() == input.size());
	sca_t value = 0;
	for (unsigned i = 0; i < truth.size(); ++i) {
		value += pow(truth[i] - input[i], 2);
	}
	return value / 2.0f;
}

vec_t __cdecl squared_err_d(const vec_t& truth, const vec_t& input) {
	return input - truth;
}

// softmax-cross entropy error function

sca_t __cdecl XE_err(const vec_t& truth, const vec_t& input) {
	vec_t softmax_input = stable_softmax(input);
	vec_t softmax_truth = stable_softmax(truth);
	return cross_entropy(softmax_truth, softmax_input);
}

vec_t __cdecl XE_err_d(const vec_t& truth, const vec_t& input) {
	return input - truth;
}

// matrix operations: matrix multiplication, with first operand transposed, with second transposed, Hadamard-multiplication
// all of them are based on the good old O(n^3) classic matrix multiplication -- veeeery slow

// p: row count of A
// q: column count of A == row count of B
// r: column count of B
// result: mtxC = mtxA*mtxB
// mtxA: p x q, mtxB: q x r, mtxC: p x r
void op_MtxMtxMul(const mtx_t& mtxA, const mtx_t& mtxB, mtx_t& mtxC, unsigned p, unsigned q, unsigned r) {
	assert(mtxA.size() == p * q); assert(mtxB.size() == q * r);
	mtxC.resize(p*r);
	for (unsigned i = 0; i < p; ++i) {
		for (unsigned j = 0; j < r; ++j) {
			mtxC[i*r + j] = 0;
			for (unsigned k = 0; k < q; ++k) {
				mtxC[i*r + j] += mtxA[i*q + k] * mtxB[k*r + j];		// mtxC[i,j] += mtxA[i,k] * mtxB[k,j]
			}
		}
	}
}

// p: COLUMN count of A (== row count of A transposed)
// q: ROW count of A (== column count of A^T == row count of B)
// r: COLUMN count of B
// mtxA: __q x p__, mtxB: q x r, mtxC: p x r
// so mtxA^T: p x r as needed
// result: mtxC = mtxA^T * mtxB (matrix multiplication with A transposed)
void op_MtxTMtxMul(const mtx_t& mtxA, const mtx_t& mtxB, mtx_t& mtxC, unsigned p, unsigned q, unsigned r) {
	assert(mtxA.size() == q * p); assert(mtxB.size() == q * r);
	mtxC.resize(p*r);
	for (unsigned i = 0; i < p; ++i) {
		for (unsigned j = 0; j < r; ++j) {
			mtxC[i*r + j] = 0;
			for (unsigned k = 0; k < q; ++k) {
				mtxC[i*r + j] += mtxA[k*p + i] * mtxB[k*r + j];		// mtxC[i,j] += mtxA[k,i] * mtxB[k,j]		<--- Note [k,i] here, not [i,k]!
			}														
		}
	}
}

// p: row count of A
// q: column count of A = row count of B transposed = column count of B,
// r: column count of B^T = row count of B
// mtxA: p x q, mtxB: r x q, mtxC: p x r
// result: mtxC = mtxA*mtxB^T (matrix multiplication with B transposed)
void op_MtxMtxTMul(const mtx_t& mtxA, const mtx_t& mtxB, mtx_t& mtxC, unsigned p, unsigned q, unsigned r) {
	assert(mtxA.size() == p * q); assert(mtxB.size() == r * q);
	mtxC.resize(p*r);
	for (unsigned i = 0; i < p; ++i) {
		for (unsigned j = 0; j < r; ++j) {
			mtxC[i*r + j] = 0;
			for (unsigned k = 0; k < q; ++k) {
				mtxC[i*r + j] += mtxA[i*q + k] * mtxB[j*q + k];		// mtxC[i,j] += mtxA[i,k] * mtxB[j,k]		<--- Note [j,k], not [k,j]!
			}
		}
	}
}

// p: row counts of A and B
// q: column counts of A and B 
// mtxA: p x q, mtxB: p x q, mtxC: p x q (all are similar)
// result: mtxC = mtxA o mtxB (Hadamard multiplication or element-wise multiplication)
void op_MtxMtxHadamard(const mtx_t& mtxA, const mtx_t& mtxB, mtx_t& mtxC, unsigned p, unsigned q) {
	mtxC.resize(p*q);
	for (unsigned i = 0; i < p; ++i) {
		for (unsigned j = 0; j < q; ++j) {
			mtxC[i*q + j] = mtxA[i*q + j] * mtxB[i*q + j];			// mtxC[i,j] = mtxA[i,j] * mtxB[i,j]
		}
	}
}

// applies scalar (vec_t -> vec_t) function to all elements of a vector (vec_t)
// valarray::apply() was harder to debug, needed own version
inline void ApplyFunc(vec_t& v, acfunc_t& fn) {
	for (unsigned i = 0; i < v.size(); ++i) {
		v[i] = fn(v[i]);
	}
}

NeuralNetwork& operator<< (NeuralNetwork& nn, unique_ptr<NNLayer>&& layer) {
	//	layer->SetMother(&nn); removed from here
	nn._layers.push_back(std::move(layer));
	return nn;
}

void Usage(const string& cmd0) {
	cout << "Usage:\n\n";
	cout << "  " << cmd0 << "  <layer1size> <activationfn1> <layer2size> <activationfn2> ... <errorfn> [<options>]" << endl;
	cout << endl;
	cout << "  first layer's size has to be " << INPUT_LAYER_SIZE << "\n";
	cout << "  number of layers, n can be arbitrary\n";
	cout << "  possible activation functions:\n";
	for (auto it : global_actFns) {
		cout << "    " << it.first << "\n";
	}
	cout << "  possible error functions:\n";
	for (auto it : global_errFns) {
		cout << "    " << it.first << "\n";
	}
	cout << "  syntax needed for options: --option=value\n";
	cout << "  possible options:\n";
	cout << "    ";
	for (auto it : global_strOpts) {
		cout << it.first << " ";
	}
	cout << "\n    ";
	for (auto it : global_floatOpts) {
		cout << it.first << " ";
	}
	cout << "\n    ";
	for (auto it : global_intOpts) {
		cout << it.first << " ";
	}
	cout << "\n  (some of them may mean the same)\n\n";
	exit(0);
}

// builds network based on the command-line parameters it's given (see also Usage())
bool BuildNetwork(NeuralNetwork & nn, std::list<string>& params) {
	cout << "----------NETWORK----------\n";
	unsigned layerno = 1;
	bool has_terminal_layer = false;
	while ((params.size() > 0) && (IsInteger(params.front()))) {
		int layersize = atoi(params.front().c_str()); 
		params.pop_front();
		cout << "layer #" << layerno << " size " << layersize << " ";
		if ((layerno == 1) && (layersize != INPUT_LAYER_SIZE)) {
			cerr << "  first layer input size has to match input dimension (" << INPUT_LAYER_SIZE << ")" << endl;
			return false;
		}
		if (params.empty()) {
			cerr << "  no activation function for layer #" << layerno << " (size: " << layersize << ")" << endl;
			return false;
		}
		string actfnname = params.front(); 
		params.pop_front();
		auto it = global_actFns.find(actfnname);
		if (it == global_actFns.end()) {
			cerr << "  unknown activation function: " << actfnname << endl;
			return false;
		}
		cout << "act.fn. " << it->first << endl;
		auto activationFunction = it->second.first;
		auto activationFnDerived = it->second.second;
		nn << std::make_unique<NNFullyConnected>(layersize);
		nn << std::make_unique<NNNonLinearity>(activationFunction, activationFnDerived);
		if (params.size() > 0) {
			// check if next argument is an error function
			auto it2 = global_errFns.find(params.front());
			if (it2 != global_errFns.end()) {
				// then this will be treated as the last argument -- pop only if this is the case
				params.pop_front(); 
				cout << "errfn. " << it2->first << endl;
				nn << std::make_unique<NNTerminal>(TERMINAL_LAYER_SIZE, it2->second.first, it2->second.second);
				has_terminal_layer = true;
				break;
			}
		}
		++layerno;
	}
	if (!has_terminal_layer) {
		cout << "  no error function given, using default (MSE)" << endl;
		nn << std::make_unique<NNTerminal>(TERMINAL_LAYER_SIZE, squared_err, squared_err_d);
	}
	nn.LinkLayerAll();
	return true;
}

// parses command-line arguments, builds the network via BuildNetwork() using them and checks for additional options in command-line
bool ParseArgs(int argc, const char* argv[], DBManager& dbm, NeuralNetwork& nn) {
	
	if (argc < 2) {
		Usage(argv[0]);
	}
	std::list<string> params;
	for (int i = 1; i < argc; ++i) {
		params.push_back(StrToUpper(argv[i]));
	}
	if (!BuildNetwork(nn, params)) {
		return false;
	}
	for (const auto& param : params) {
		if (param.substr(0, 2) != "--") {
			cerr << "  unknown parameter ignored: " << param << endl;
			continue;
		}
		auto sep = param.find('=');
		if (sep == string::npos) {
			cerr << "  unknown parameter ignored: " << param << endl;
			continue;
		}
		string pname = param.substr(2, sep - 2); 
		string pval = param.substr(sep + 1, string::npos); 
		{
			auto it = global_strOpts.find(pname);
			if (it != global_strOpts.end()) {
				*(it->second) = pval;
				continue;
			}
		}
		{
			auto it = global_intOpts.find(pname);
			if (it != global_intOpts.end()) {
				if (!IsInteger(pval)) {
					cerr << "  " << pname << " option has to be an integer" << endl;
					return false;
				}
				else {
					*(it->second) = atoi(pval.c_str());
				}
				continue;
			}
		}
		{
			auto it = global_floatOpts.find(pname);
			if (it != global_floatOpts.end()) {
				if (!IsFloat(pval)) {
					cerr << "  " << pname << " option has to be a number" << endl;
					return false;
				}
				*(it->second) = static_cast<sca_t>(atof(pval.c_str()));
				continue;
			}
		}
		cerr << "  unknown option ignored: " << pname << endl;
	} // for i
	return true;
}

void PrintParams() {
	vector<string> p = { "LR", "BATCHSIZE", "TRUNCDATASIZE", "NORMALIZE", "PASSES", "MTXEV", "MTXMEAN", "MTXD", "MTXDEV", "CURVE", "SHOWCURVETRUNC" };
	cout << "----------PARAMS----------\n";
	for (auto it : p) {
		auto it1 = global_strOpts.find(it);
		if (it1 != global_strOpts.end()) {
			cout << it << ": " << *it1->second << endl;
			continue;
		}
		auto it2 = global_intOpts.find(it);
		if (it2 != global_intOpts.end()) {
			cout << it << ": " << *it2->second << endl;
			continue;
		}
		auto it3 = global_floatOpts.find(it);
		if (it3 != global_floatOpts.end()) {
			cout << it << ": " << *it3->second << endl;
		}
	}
}

bool IsValidAcFunc(const acfunc_t& func) {
	for (auto it : global_actFns) {
		if (it.second.first == func) return true;
		if (it.second.second == func) return true;
	}
	return false;
}
bool IsValidErFunc(const erfunc_t& func) {
	for (auto it : global_errFns) {
		if (it.second.first == func) return true;
	}
	return false;
}
bool IsValidErDFunc(const erfuncd_t& func) {
	for (auto it : global_errFns) {
		if (it.second.second == func) return true;
	}
	return false;
}

// NNFullyConnected class

// Initializes weight matrix and bias vector
// (needs not to be called in ctor, since we don't know there the output dimension)
void NNFullyConnected::InitMatrix(unsigned n, unsigned m)
{
	_weightmatrix_rows = n;
	_weightmatrix_cols = m;
#ifdef USE_WEIGHT_NORMALIZATION
	_weightmatrix_v.resize(n*m);
	_weightupdate_v.resize(n*m);
	_weightmatrix_g.resize(n);
	_weightupdate_g.resize(n);

	_weightupdate_v = 0;
	_weightupdate_g = 0;
#else
	_weightmatrix.resize(n*m);
	_weightupdate.resize(n*m);
	_weightupdate = 0;
#endif
	_bias.resize(_weightmatrix_rows);
	_biasupdate.resize(_weightmatrix_rows);
	_biasupdate = 0;
	
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(global_mtxmean, global_mtxdev);

#ifdef USE_WEIGHT_NORMALIZATION
	for (unsigned i = 0; i < _weightmatrix_v.size(); ++i) {
		_weightmatrix_v[i] = static_cast<sca_t>(distribution(generator));
	}
	for (unsigned i = 0; i < _weightmatrix_g.size(); ++i) {
		_weightmatrix_g[i] = static_cast<sca_t>(distribution(generator));
	}
#else
	for (unsigned i = 0; i < _weightmatrix.size(); ++i) {
		_weightmatrix[i] = static_cast<sca_t>(distribution(generator));
	}
#endif
	
	for (unsigned i = 0; i < _bias.size(); ++i) {
		_bias[i] = static_cast<sca_t>(distribution(generator));
	}
}

void NNFullyConnected::InitBatch() {
#ifdef USE_WEIGHT_NORMALIZATION
	_weightupdate_v = 0;
	_weightupdate_g = 0;
#else
	_weightupdate = 0;
#endif
	_biasupdate = 0;
	_batchcnt = 0;
}

// Checks if the layer is fully constructed, so feed can be started
// _input,_output not needed for "Valid" state
bool NNFullyConnected::IsValid() const {
	return ((_mothernetwork != nullptr) &&
		(_batchcnt == 0) &&
		(_inputsize > 0) &&
		(_outputsize > 0) &&
		(_weightmatrix_cols == _inputsize) &&
		(_weightmatrix_rows == _outputsize) &&
		(_learningrate > 0) &&
#ifdef USE_WEIGHT_NORMALIZATION
		(_weightmatrix_v.size() == _weightmatrix_cols * _weightmatrix_rows) &&
		(_weightupdate_v.size() == _weightmatrix_v.size()) &&
		(_weightmatrix_g.size() == _weightmatrix_rows) &&
		(_weightupdate_g.size() == _weightmatrix_g.size()) &&
#else
		(_weightmatrix.size() == _weightmatrix_cols * _weightmatrix_rows) &&
		(_weightupdate.size() == _weightmatrix.size()) &&
#endif
		(_bias.size() == _weightmatrix_rows) &&
		(_biasupdate.size() == _weightmatrix_rows));
}

bool NNNonLinearity::IsValid() const
{
	return (_mothernetwork != nullptr) &&
		(_inputsize > 0) &&
		(_outputsize > 0) &&
		IsValidAcFunc(_activFn) && 
		IsValidAcFunc(_activFnDerived);
}

void NNFullyConnected::LinkLayer(NeuralNetwork* mothernetwork, unsigned nextLayerSize) {
	_mothernetwork = mothernetwork;

	// When NeuralNetwork::_layers gets the next layer, this is the point 
	// when the output dimension of the previous layer is determined (= the
	// input dimension of the next layer). 

	_outputsize = nextLayerSize;
	InitMatrix(_outputsize, _inputsize);
}

void NNNonLinearity::LinkLayer(NeuralNetwork* mothernetwork, unsigned nextLayerSize) {
	_mothernetwork = mothernetwork;

	_inputsize = nextLayerSize;
	_outputsize = nextLayerSize;
}

#ifdef USE_WEIGHT_NORMALIZATION
// see [1] eq (2), difference: in (2) g is scalar, v,w are vectors, here g is a vector, v,w are matrices - we apply (2) row-by-row
mtx_t NNFullyConnected::CalcWeightMatrixFromVandG() const
{
	mtx_t w(_weightmatrix_v.size());		// _weightmatrix_rows*_weightmatrix_cols
	w = 0;
	for (size_t i = 0; i < _weightmatrix_rows; ++i)
	{
		const size_t row_len = _weightmatrix_cols;
		sca_t const* v_row_vec = &_weightmatrix_v[i * row_len];
		sca_t* w_row_vec = &w[i * row_len];

		sca_t row_norm = 0;
		for (size_t j = 0; j < row_len; ++j)
		{
			row_norm += pow(v_row_vec[j], 2.0);
		}
		row_norm = sqrt(row_norm);

		for (size_t j = 0; j < row_len; ++j)
		{
			w_row_vec[j] = v_row_vec[j] / row_norm * _weightmatrix_g[i];
		}
	}
	return w;
}

// see [1] eq (3) - we apply (3) for each row vector of w,v, each component of g
void NNFullyConnected::CalcGandVupdateFromWupdate(const mtx_t& wu, mtx_t& vu, vec_t& gu) const
{
	gu.resize(_weightmatrix_g.size());		// _weightmatrix_rows
	vu.resize(_weightmatrix_v.size());		// _weightmatrix_rows*_weightmatrix_cols
	for (size_t i = 0; i < _weightmatrix_rows; ++i)
	{
		const size_t row_len = _weightmatrix_cols;
		sca_t const* v_row_vec = &_weightmatrix_v[i * row_len];
		sca_t const* wu_row_vec = &wu[i * row_len];
		sca_t* vu_row_vec = &vu[i * row_len];

		// calculate ith component of gu

		sca_t row_norm_sq = 0;
		for (size_t j = 0; j < row_len; ++j)
		{
			row_norm_sq += pow(v_row_vec[j], 2.0);
		}
		const sca_t row_norm = sqrt(row_norm_sq);

		sca_t inner_prod = 0;
		for (size_t j = 0; j < row_len; ++j)
		{
			inner_prod += v_row_vec[j] * wu_row_vec[j];
		}
		gu[i] = inner_prod / row_norm;

		// calculate ith row of vu

		for (size_t j = 0; j < row_len; ++j)
		{
			const sca_t minuend = _weightmatrix_g[i] / row_norm * wu_row_vec[j];
			const sca_t subtrahend = _weightmatrix_g[i] * gu[i] / row_norm_sq * v_row_vec[j];
			vu_row_vec[j] = minuend - subtrahend;
		}
	}
}
#endif//USE_WEIGHT_NORMALIZATION

vec_t NNNonLinearity::FwdProp(vec_t& input)
{
	_input = input;
	_output = _input;
	ApplyFunc(_output, _activFn);
	return _output;
}

vec_t NNNonLinearity::BackProp(vec_t& Wdelta)
{
	vec_t actFnDerAppliedToInput = _input;															// was: _cached_preout in NNFullyConnected::BackProp()
	ApplyFunc(actFnDerAppliedToInput, _activFnDerived);
	vec_t delta_to_pass;																			// was: mydelta in NNFullyConnected::BackProp()
	op_MtxMtxHadamard(Wdelta, actFnDerAppliedToInput, delta_to_pass, 1, Wdelta.size());
	return delta_to_pass;
}

vec_t NNFullyConnected::FwdProp(vec_t& input)
{
	vec_t output;
	_input = input;

#ifdef USE_WEIGHT_NORMALIZATION
	mtx_t temp_weightmatrix = CalcWeightMatrixFromVandG();
	op_MtxMtxMul(temp_weightmatrix, _input, output, _weightmatrix_rows, _weightmatrix_cols, 1);
#else
	op_MtxMtxMul(_weightmatrix, _input, output, _weightmatrix_rows, _weightmatrix_cols, 1);
#endif

	output += _bias;
	_output = output;
	return output;
}

vec_t NNFullyConnected::BackProp(vec_t & Wdelta)
{
	vec_t delta_to_pass;
#ifdef USE_WEIGHT_NORMALIZATION
	mtx_t temp_weightmatrix = CalcWeightMatrixFromVandG();
	op_MtxTMtxMul(temp_weightmatrix, Wdelta, delta_to_pass, _weightmatrix_cols, _weightmatrix_rows, 1);
#else
	op_MtxTMtxMul(_weightmatrix, Wdelta, delta_to_pass, _weightmatrix_cols, _weightmatrix_rows, 1);
#endif
		// W^T * delta

	mtx_t wu;
	op_MtxMtxTMul(Wdelta, _input, wu, Wdelta.size(), 1, _input.size());
		// delta * _input^T

#ifdef USE_WEIGHT_NORMALIZATION
	mtx_t vu;
	vec_t gu;
	CalcGandVupdateFromWupdate(wu, vu, gu);
	_weightupdate_v -= vu;
	_weightupdate_g -= gu;
#else
	_weightupdate -= wu;
#endif

	_biasupdate -= Wdelta;
	_batchcnt++;
	return delta_to_pass;
}

void NNFullyConnected::UpdateWeightMatrix() {
#ifdef USE_WEIGHT_NORMALIZATION
	_weightmatrix_v += _learningrate * _weightupdate_v;
	_weightmatrix_g += _learningrate * _weightupdate_g;
#else
	_weightmatrix += _learningrate * _weightupdate;
#endif
	_bias += _learningrate * _biasupdate;
}

// DBManager class

// Reading a uint32_t starting from "bufseek" (steps bufseek by sizeof(uint32_t))
// Note: couldn't use reinterpret_cast<uint32_t*> since some platforms are
// MSB first, some are LSB first.
inline uint32_t DBManager::ReadUint32MSBF(unsigned char*& bufseek) {
	uint32_t value =
		(bufseek[0] << 24) +
		(bufseek[1] << 16) +
		(bufseek[2] << 8) +
		bufseek[3];
	bufseek += 4;
	return value;
}

bool DBManager::ReadDB(const string & fn_images, const string & fn_labels, uint32_t magic_images, uint32_t magic_labels, vector<vec_t>& images, vector<sca_t>& labels)
{
	typedef std::ifstream inputstream;
	std::unique_ptr<unsigned char[]> mainbuffer;
	cout << "Reading database (" << fn_images << ")...";
	{	// fn_images file --> mainbuffer
		inputstream is(fn_images, std::ios::binary | std::ios::ate | std::ios::in);
		if (!is) {
			cout << "  Error opening database file!" << endl;
			return false;
		}
		auto fileSize = static_cast<size_t>(is.tellg());
		is.seekg(0, std::ios::beg);
		if (fileSize == 0) {
			cout << "  Error: database file seems empty (filesize=0)" << endl;
			return false;
		}
		try {
			mainbuffer = std::make_unique<unsigned char[]>(fileSize);
		}
		catch (const std::bad_alloc& e) {
			cout << "  Memory allocation error (out of memory?) [" << e.what() << "]" << endl;
			return false;
		}
		is.read(reinterpret_cast<char*>(mainbuffer.get()), fileSize);
		cout << "OK\n";
		cout << "  Filesize: " << fileSize << "\n";
	}
	unsigned char* bufseekptr = mainbuffer.get();
	{	// mainbuffer -> images
		auto temp = ReadUint32MSBF(bufseekptr);
		if (temp != magic_images) {
			cerr << "MAGIC number mismatch: " << temp << "!=" << magic_images << endl;
			return false;
		}
		cout << "  Magic [" << temp << "]=[" << magic_images << "]\n";
		auto expected_elements = std::min(global_truncdatasize, ReadUint32MSBF(bufseekptr)); 
		cout << "  # of elements: " << expected_elements << "\n";
		auto rows = ReadUint32MSBF(bufseekptr);
		auto cols = ReadUint32MSBF(bufseekptr);
		cout << "  Size: " << rows << "x" << cols << "\n";
		images.reserve(expected_elements);
		for (unsigned i = 0; i < expected_elements; ++i) {
			images.emplace_back(rows*cols);
			auto& oneimg = images.back();
			for (unsigned j = 0; j < rows*cols; ++j) {
				oneimg[j] = static_cast<sca_t>(*bufseekptr++);		// loading uint8_t data into sca_t vectors
			}
		}
	}
	cout << "Reading labels file (" << fn_labels << ")...";
	{	// fn_labels file --> mainbuffer
		inputstream is(fn_labels, std::ios::binary | std::ios::ate | std::ios::in);
		if (!is) {
			cout << "  Error opening labels file!" << endl;
			return false;
		}
		int fileSize = static_cast<int>(is.tellg());
		is.seekg(0, std::ios::beg);
		if (fileSize == 0) {
			cout << "  Error: labels file seems empty (filesize=0)" << endl;
			return false;
		}
		try {
			mainbuffer.reset(new unsigned char[fileSize]);
		}
		catch (const std::bad_alloc& e) {
			cout << "  Memory allocation error (out of memory?) [" << e.what() << "]" << endl;
			return false;
		}
		is.read(reinterpret_cast<char*>(mainbuffer.get()), fileSize);
		cout << "OK\n";
		cout << "  Filesize: " << fileSize << "\n";
	}
	bufseekptr = mainbuffer.get();
	{	// mainbuffer -> labels
		auto temp = ReadUint32MSBF(bufseekptr);
		if (temp != magic_labels) {
			cerr << "MAGIC number mismatch: " << temp << "!=" << magic_labels << endl;
			return false;
		}
		cout << "  Magic [" << temp << "]=[" << magic_labels << "]\n";
		auto expected_elements = std::min(global_truncdatasize, ReadUint32MSBF(bufseekptr));
		cout << "  # of elements: " << expected_elements << "\n";
		labels.reserve(expected_elements);
		for (unsigned i = 0; i < expected_elements; ++i) {
			labels.push_back(static_cast<sca_t>(*bufseekptr++));	// loading uint8_t data into sca_t vectors
		}
	}
	cout << "Read OK\n";
	return true;
}

void DBManager::CalcStatParams(vector<vec_t>& data, sca_t& mean, sca_t& dev) {
	double s = 0;
	double m = 0;
	unsigned count = 0;
	for (unsigned i = 0; i < data.size(); ++i) {
		for (unsigned j = 0; j < data[i].size(); ++j) {
			m += data[i][j];
			count++;
		}
	}
	m /= count;

	for (unsigned i = 0; i < data.size(); ++i) {
		for (unsigned j = 0; j < data[i].size(); ++j) {
			s += pow(data[i][j] - m, 2);
		}
	}

	s = sqrt(s / (count - 1));

	mean = static_cast<sca_t>(m); dev = static_cast<sca_t>(s);
}

void DBManager::NormalizeDB(vector<vec_t>& data, const sca_t& mean, const sca_t& dev) {
	for (unsigned i = 0; i < data.size(); ++i) {
		for (unsigned j = 0; j < data[i].size(); ++j) {
			data[i][j] -= mean;
			data[i][j] /= dev;
		}
	}
}

void DBManager::PrintMatrixToConsole(const mtx_t& mtx, unsigned rows, unsigned cols) {
	cerr << std::fixed << std::setw(4) << std::setprecision(1);
	for (unsigned i = 0; i < rows; ++i) {
		for (unsigned j = 0; j < cols; ++j) {
			cerr << mtx[i*cols + j] << " ";
		}
		cerr << endl;
	}
}

void DBManager::DrawOneLetterToConsole(bool testOrTRAIN, int nr, int threshold) {
	for (unsigned i = 0; i < 28; ++i) {
		for (unsigned j = 0; j < 28; ++j) {
			if (testOrTRAIN) {
				cerr << (trainimages[nr][i * 28 + j] > threshold ? "X" : " ");
			}
			else {
				cerr << (testimages[nr][i * 28 + j] > threshold ? "X" : " ");
			}
		}
		cerr << endl;
	}
	if (testOrTRAIN) {
		cerr << trainlabels[nr] << endl;
	}
	else {
		cerr << testlabels[nr] << endl;
	}
}

inline void one_hot_convert(const sca_t& val, vec_t& one_hot) {
	assert(one_hot.size() == 10 && val >= 0 && val <= 9);
	one_hot = 0;
	one_hot[static_cast<unsigned>(round(val))] = 1;
}

void DBManager::FeedDatabase(const vector<vec_t>& database, const vector<sca_t>& truthdb, NeuralNetwork& neuralnet, const unsigned& batchSize, const sca_t& learningRate)
{
	assert(database.size() == truthdb.size());
	neuralnet.SetLearningRate(learningRate);
	for (unsigned passes_cnt = 0; passes_cnt < global_passes; ++passes_cnt) {
		cout << "PASS " << passes_cnt + 1 << endl;
		auto image_cit = database.cbegin();
		auto truth_cit = truthdb.cbegin();
		vec_t truth_one_hot = vec_t(10);
		vec_t truth_scalar = vec_t(1);
		NNTerminal& terminalLayer = dynamic_cast<NNTerminal&>(*(neuralnet.GetLastLayer()));
		while (image_cit != database.cend()) {
			neuralnet.InitBatchAll();
			for (unsigned batchCnt = 0;
				(batchCnt < batchSize) && (image_cit != database.cend()) && (truth_cit != truthdb.cend());
				++batchCnt, ++image_cit, ++truth_cit) {

				neuralnet.FwdPropThrough(*image_cit);
				if (global_truth_conv10_one_hot_conversion) {
					one_hot_convert(*truth_cit, truth_one_hot);
					terminalLayer.setTruth(truth_one_hot);
				}
				else {
					truth_scalar[0] = *truth_cit;
					terminalLayer.setTruth(truth_scalar);
				}
				neuralnet.BackPropThrough();
			} // batch
			cerr << "."; // one dot for every batch
			neuralnet.UpdateWeightsThrough();
			if (global_showcurve_enabled) {
				TestDatabase(database, truthdb, neuralnet, global_showcurve_trunc, true);
			}
			if (global_matrixcurve_fn != "") {
				cout << "S\n";
				SaveMatrix(global_matrixcurve_fn, neuralnet, true);
			}
		} // database
		cerr << endl;
	} // pass
}

void DBManager::TestDatabase(const vector<vec_t>& database, const vector<sca_t>& truthdb, NeuralNetwork & neuralnet, unsigned showcurve_trunc, bool because_showcurve)
{
	assert(database.size() == truthdb.size());
	if (showcurve_trunc == 0) showcurve_trunc = database.size();
	auto image_cit = database.cbegin();
	auto truth_cit = truthdb.cbegin();
	vec_t truth_one_hot = vec_t(10);
	vec_t truth_scalar = vec_t(1);
	NNTerminal& terminalLayer = dynamic_cast<NNTerminal&>(*(neuralnet.GetLastLayer()));
	unsigned test_all = 0;
	unsigned test_hit = 0;
	sca_t sum_err = 0;
	while ((image_cit != database.cend()) && (truth_cit != truthdb.cend())) {
		neuralnet.FwdPropThrough(*image_cit);
		if (global_truth_conv10_one_hot_conversion) {
			one_hot_convert(*truth_cit, truth_one_hot);
			terminalLayer.setTruth(truth_one_hot);
		}
		else {
			truth_scalar[0] = *truth_cit;
			terminalLayer.setTruth(truth_scalar);
		}
		sum_err += terminalLayer.GetResult();
		test_hit += terminalLayer.IsAccurate();
		++image_cit, ++truth_cit, ++test_all;
		if (!because_showcurve && (test_all % global_batchsize == 0)) cerr << ".";
		if (because_showcurve && (test_all >= showcurve_trunc)) break;
	}
	if (!because_showcurve) {
		cout << "\nAccuracy: " << test_hit << "/" << test_all << "=" << static_cast<float>(test_hit) / test_all * 100 << "%" << endl;
		cout << "Total error: " << sum_err << endl;
	}
	else {
		cout << " " << static_cast<float>(test_hit) / test_all * 100 << "% (" << sum_err << ")" << endl;
	}
}

// CLASS: NeuralNetwork

void NeuralNetwork::FwdPropThrough(const vec_t & initial_input)
{
	vec_t input = initial_input;
	for (auto it = _layers.begin(); it != _layers.end(); ++it) {
		input = it->get()->FwdProp(input);
	}
}

void NeuralNetwork::BackPropThrough()
{
	vec_t delta;
	for (auto it = _layers.rbegin(); it != _layers.rend(); ++it) {
		delta = it->get()->BackProp(delta);
	}
}

void NeuralNetwork::UpdateWeightsThrough()
{
	for (auto it = _layers.rbegin(); it != _layers.rend(); ++it) {
		it->get()->UpdateWeightMatrix();
	}
}

bool NeuralNetwork::IsValid() const
{
	if ((_layers.size() == 0) || (dynamic_cast<NNTerminal*>(_layers.back().get()) == nullptr)) {
		return false;
	}
	bool value = true;
	for (auto cit = _layers.cbegin(); cit != _layers.cend(); ++cit) {
		value = value && cit->get()->IsValid();
	}
	return value;
}

void NeuralNetwork::SetLearningRate(const sca_t & learningrate)
{
	for (auto cit = _layers.cbegin(); cit != _layers.cend(); ++cit) {
		cit->get()->SetLearningRate(learningrate);
	}
}

void NeuralNetwork::InitBatchAll()
{
	for (auto cit = _layers.cbegin(); cit != _layers.cend(); ++cit) {
		cit->get()->InitBatch();
	}
}

void NeuralNetwork::LinkLayerAll()
{
	auto terminalLayer = _layers.rbegin()->get();
	terminalLayer->LinkLayer(this, 0);								// handle terminalLayer differently

	auto rit = ++_layers.rbegin();
	auto nextLayerRit = _layers.rbegin();
	for (; rit != _layers.rend(); ++rit, ++nextLayerRit)
	{
		unsigned nextLayerSize = TERMINAL_LAYER_SIZE;
		if (nextLayerRit != _layers.rbegin())
			nextLayerSize = nextLayerRit->get()->GetInputSize();
		rit->get()->LinkLayer(this, nextLayerSize);
	}
}

// CLASS: NNTerminal

void NNTerminal::LinkLayer(NeuralNetwork * mothernetwork, unsigned /* nextLayerSize unused */ ) {
	_mothernetwork = mothernetwork;
}

// In the last (pseudo) layer, forward propagation means only to 
// store the given input, so 
// input field of NNTerminal layer = output of whole neural network
vec_t NNTerminal::FwdProp(vec_t & input) {	
	_input = input;
	vec_t dummy;
	return dummy;	// unused at caller
}

// (function argument (otherwise named Wdelta) is unused here)
vec_t NNTerminal::BackProp(vec_t &) {
	return _errorFnDerived(_truth, _input); 
}

// _truth, _input: not needed for "Valid" state
bool NNTerminal::IsValid() const {
	return ((_mothernetwork != nullptr) &&
		(_inputsize > 0) &&
		IsValidErFunc(_errorFn) &&
		IsValidErDFunc(_errorFnDerived));
}

// Error of the neural network produced result, measured with the selected error function
// (Only meaningful if NNTerminal::SetTruth() has already been called.)
// called by DBManager::TestDatabase()
sca_t NNTerminal::GetResult() const {
	return _errorFn(_truth, _input);
}

// Returns 1 if the neural network produced the correct result/label ("argmax(output)=label"), 
// 0 otherwise.
// (Only meaningful if NNTerminal::SetTruth() has already been called.)
// called by DBManager::TestDatabase()
unsigned NNTerminal::IsAccurate() const {
	assert(_input.size() == _truth.size());
	sca_t inputmax = -INFINITY;
	sca_t truthmax = -INFINITY;
	unsigned inputargmax = 0;
	unsigned truthargmax = 0;
	for (unsigned i = 0; i < _input.size(); ++i) {
		if (_input[i] > inputmax) {
			inputmax = _input[i];
			inputargmax = i;
		}
		if (_truth[i] > truthmax) {
			truthmax = _truth[i];
			truthargmax = i;
		}
	}
	if (global_truth_conv10_one_hot_conversion) {
		return (inputargmax == truthargmax ? 1 : 0);
	}
	else {
		return (inputargmax == _truth[0] ? 1 : 0);
	}
}

bool LoadMatrix(const string& fn, NeuralNetwork& nn) {
	cout << "Loading matrix (" << fn << ")...";
	{
		std::ifstream is(fn, std::ios::in);
		if (!is) {
			cout << "  Error opening matrix file!" << endl;
			return false;
		}
		
		unsigned nr_layers;
		is >> nr_layers;
		if (nr_layers != nn._layers.size()) {
			cerr << "LoadMatrix nr_size mismatch " << nr_layers << "!=" << nn._layers.size() << endl;
			return false;
		}
		for (auto it = nn._layers.begin(); it != nn._layers.end(); ++it) {
			NNFullyConnected& fcl = dynamic_cast<NNFullyConnected&>(*(it->get()));
			unsigned isize, osize;
			is >> isize >> osize;
			if (fcl._inputsize != isize || fcl._outputsize != osize) {
				cerr << "LoadMatrix layer size mismatch " << isize << "x" << osize << "!=" << fcl._inputsize << "x" << fcl._outputsize << endl;
				return false;
			} // if size mismatch
		} // for it
		for (auto it = nn._layers.begin(); it != nn._layers.end(); ++it) {
			NNFullyConnected& fcl = dynamic_cast<NNFullyConnected&>(*(it->get()));
#ifdef USE_WEIGHT_NORMALIZATION
			for (unsigned idx = 0; idx < fcl._inputsize * fcl._outputsize; ++idx) {
				is >> fcl._weightmatrix_v[idx];
			}
			for (unsigned idx = 0; idx < fcl._outputsize; ++idx) {
				is >> fcl._weightmatrix_g[idx];
			}
#else
			for (unsigned idx = 0; idx < fcl._inputsize * fcl._outputsize; ++idx) {
				is >> fcl._weightmatrix[idx];
			}
#endif
			for (unsigned idx = 0; idx < fcl._outputsize; ++idx) {
				is >> fcl._bias[idx];
			}
		} // for it
	} // with ifstream
	cout << "OK\n";
	return true;
}

bool SaveMatrix(const string& fn, NeuralNetwork& nn, bool because_mtxcurve = false) {
	if (!because_mtxcurve) {
		cout << "Saving matrix (" << fn << ")...";
	}
	static unsigned callcnt = 0; ++callcnt;
	{
		assert(fn != "");
		std::ofstream os(fn, (because_mtxcurve ? std::ios::out | std::ios::app : std::ios::out) );			
		if (because_mtxcurve) os << "CALLCNT " << callcnt << endl;
		os << nn._layers.size() << " ";
		for (auto it = nn._layers.begin(); it != nn._layers.end(); ++it) {
			NNFullyConnected& fcl = dynamic_cast<NNFullyConnected&>(*(it->get()));
			os << fcl._inputsize << " " << fcl._outputsize << " ";
		}
		os << endl;
		for (auto it = nn._layers.begin(); it != nn._layers.end(); ++it) {
			NNFullyConnected& fcl = dynamic_cast<NNFullyConnected&>(*(it->get()));
			if (because_mtxcurve) os << "LAYER " << it - nn._layers.begin() << endl;
#ifdef USE_WEIGHT_NORMALIZATION
			unsigned wmsize_v = fcl._weightmatrix_cols * fcl._weightmatrix_rows;
			for (unsigned i = 0; i < wmsize_v; ++i) {
				os << fcl._weightmatrix_v[i] << ((i + 1) % fcl._weightmatrix_cols == 0 ? "\n" : " ");
			} // for i
			os << "\n\n";
			unsigned wmsize_g = fcl._weightmatrix_rows;
			for (unsigned i = 0; i < wmsize_g; ++i) {
				os << fcl._weightmatrix_g[i] << ((i + 1) % fcl._weightmatrix_cols == 0 ? "\n" : " ");
			} // for i
			os << "\n\n";
#else
			unsigned wmsize = fcl._weightmatrix_cols * fcl._weightmatrix_rows;
			for (unsigned i = 0; i < wmsize; ++i) {
				os << fcl._weightmatrix[i] << ((i + 1) % fcl._weightmatrix_cols == 0 ? "\n" : " ");
			} // for i
			os << "\n\n";
#endif
			unsigned biassize = fcl._outputsize;
			for (unsigned i = 0; i < biassize; ++i) {
				os << fcl._bias[i] << " ";
			}
			os << "\n\n";
		} // for it
		os.close();
	}
	if (!because_mtxcurve) cout << "OK\n";
	return true;
}

void InitGlobalArrays() {
	global_actFns["SIGM"] = std::make_pair(stable_sigmoid, sigmoid_d);
	global_actFns["ID"] = std::make_pair(identity, identity_d);
	global_actFns["TANH"] = std::make_pair(ac_tanh, ac_tanh_d);
	global_actFns["RELU"] = std::make_pair(relu, relu_d);
	global_actFns["SRELU"] = std::make_pair(srelu, srelu_d);
	global_errFns["MSE"] = std::make_pair(squared_err, squared_err_d);
	global_errFns["XE"] = std::make_pair(XE_err, XE_err_d);
	global_strOpts["TRAINIMAGES"] = &global_trainimages_fn;
	global_strOpts["TRAINIMAGE"] = &global_trainimages_fn; 
	global_strOpts["TRAINLABELS"] = &global_trainlabels_fn;
	global_strOpts["TESTIMAGES"] = &global_testimages_fn;
	global_strOpts["TESTLABELS"] = &global_testlabels_fn;
	global_strOpts["TRAINLABEL"] = &global_trainlabels_fn;
	global_strOpts["TESTIMAGE"] = &global_testimages_fn;
	global_strOpts["TESTLABEL"] = &global_testlabels_fn;
	global_intOpts["PASS"] = &global_passes;
	global_intOpts["PASSES"] = &global_passes;
	global_intOpts["TRUNCDATASIZE"] = &global_truncdatasize;
	global_intOpts["TRUNCATE"] = &global_truncdatasize;
	global_intOpts["TRUNC"] = &global_truncdatasize;
	global_intOpts["BATCH"] = &global_batchsize;
	global_intOpts["BATCHSIZE"] = &global_batchsize;
	global_intOpts["NORMALIZE"] = &global_normalize_enabled;
	global_intOpts["NORM"] = &global_normalize_enabled;
	global_intOpts["CURVE"] = &global_showcurve_enabled;
	global_intOpts["SHOWCURVE"] = &global_showcurve_enabled;
	global_intOpts["CURVETRUNC"] = &global_showcurve_trunc;
	global_intOpts["SHOWCURVETRUNC"] = &global_showcurve_trunc;
	global_floatOpts["LEARNINGRATE"] = &global_learningrate;
	global_floatOpts["LEARN"] = &global_learningrate;
	global_floatOpts["LR"] = &global_learningrate;
	global_strOpts["USEMATRIX"] = global_strOpts["USEMTX"] = global_strOpts["LOADSAVEMATRIX"] = &global_loadsavematrix_fn;
	global_strOpts["SAVEMTX"] = global_strOpts["SAVEMATRIX"] = &global_savematrix_fn;
	global_strOpts["LOADMTX"] = global_strOpts["LOADMATRIX"] = &global_loadmatrix_fn;
	global_strOpts["MTXCURVE"] = global_strOpts["MATRIXCURVE"] = &global_matrixcurve_fn;
	global_floatOpts["MTXEV"] = &global_mtxmean;
	global_floatOpts["MTXMEAN"] = &global_mtxmean;
	global_floatOpts["MTXD"] = &global_mtxdev;
	global_floatOpts["MTXDEV"] = &global_mtxdev;
}

int main(int argc, const char* argv[])
{
	InitGlobalArrays();

	DBManager dbm;
	NeuralNetwork nn;

	constexpr bool overrideCommandlineArgs = false;
	const char** av = nullptr;
	int ac = 0;
	if (overrideCommandlineArgs)
	{
		const char *av_test[] = { argv[0], 
			"784", 
			"relu", 
			"300", 
			"id", 
			"xe", 
			"--batch=10", 
			"--trunc=10", 
			"--learningrate=0.001", 
			"--curve=1", 
			"--CURVETRUNC=10",
			"--loadsavematrix=wmatrix.mtx",
//			"--matrixcurve=mtx.log",
			"--trainimages=train-images.idx3-ubyte",
			"--trainlabels=train-labels.idx1-ubyte",
			"--testimages=t10k-images.idx3-ubyte",
			"--testlabels=t10k-labels.idx1-ubyte",
			nullptr };
			
		av = av_test;
		ac = sizeof(av_test)/sizeof(char*) - 1;
	}
	else
	{
		av = argv;
		ac = argc;
	}

	if (!ParseArgs(ac, av, dbm, nn)) {
		exit(-1);
	}

	if (!dbm.ReadDB(global_trainimages_fn, global_trainlabels_fn, TRAINIMAGES_MAGIC, TRAINLABELS_MAGIC, dbm.trainimages, dbm.trainlabels)) {
		exit(-1);
	}
	if (!dbm.ReadDB(global_testimages_fn, global_testlabels_fn, TESTIMAGES_MAGIC, TESTLABELS_MAGIC, dbm.testimages, dbm.testlabels)) {
		exit(-1);
	}

	PrintParams();

	cout << "Calculating mean & dev...";
	sca_t mean, dev;
	dbm.CalcStatParams(dbm.trainimages, mean, dev);

	cout << "\nMean: " << mean << "\nDev: " << dev << endl;

	if (global_normalize_enabled) {
		cout << "Normalizing...";
		dbm.NormalizeDB(dbm.trainimages, mean, dev);
	}

	cout << "\nValidating network...";

	nn.InitBatchAll();
	if (!nn.IsValid()) {
		cout << "ERROR: INVALID NETWORK" << endl;
		exit(-1);
	}
	else {
		cout << "OK" << endl;
	}

	{
		string mtxfn = (global_loadsavematrix_fn != "" ? global_loadsavematrix_fn : global_loadmatrix_fn);
		if (mtxfn != "") {
			if (!LoadMatrix(mtxfn, nn)) {
				exit(-1);
			}
		}
	} // with mtxfn

	cout << "\nFeeding data...\n";

	dbm.FeedDatabase(dbm.trainimages, dbm.trainlabels, nn, global_batchsize, global_learningrate);

	if (global_normalize_enabled) {
		cout << "Normalizing...";
		dbm.NormalizeDB(dbm.testimages, mean, dev);
	}

	cout << "\nTesting model on train dataset...\n";

	dbm.TestDatabase(dbm.trainimages, dbm.trainlabels, nn);

	cout << "\nTesting model on test dataset...\n";

	dbm.TestDatabase(dbm.testimages, dbm.testlabels, nn);

	{
		string mtxfn = (global_loadsavematrix_fn != "" ? global_loadsavematrix_fn : global_savematrix_fn);
		if (mtxfn != "") SaveMatrix(mtxfn, nn, false);
	}

	cout << "-------------------------------" << endl;

	return 0;
}