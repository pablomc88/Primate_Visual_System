/*
 *  AII_amacrine.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef AII_AMACRINE_H
#define AII_AMACRINE_H

// C includes:
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>

// Includes from nestkernel:
#include "archiving_node.h"
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"


// Includes from sli:
#include "dictdatum.h"

namespace mynest
{

/*
This file is part of the project published in [1,2]. The software is licensed
under the GNU General Public License. You should have received a copy of the
GNU General Public License along with the source code.

BeginDocumentation

Name: AII_amacrine - Non-spiking neuron with gap-junction currents

Description:

Implementation that is similar to parvo_neuron but with gap-junction currents
instead of conductance-based currents.

The dynamics of the neuron are defined by:

dV_m/dt = - g_L * ( V_m - E_L ) / C_m - I_gap(t) / C_m + I_e / C_m

where I_gap are the gap junction currents entering the cell:

I_gap(t) = Sum[g_gap (V_m - V_pre_i) for i in number of input electrical synapses]

g_gap is the gap junction conductance and V_pre_i the presynaptic membrane
potentials of the connected cells.

V_pre_i is calculated from the value of the synaptic conductance g_i of the
pre-synaptic cell (sent within a CurrentEvent). This conductance is described by
an instantaneous sigmoid function. V_pre_i is cleared from the sigmoid function
as follows:

V_pre_i = theta_syn - k_syn Ln(1/g_i - 1)

Parameters:
The following parameters can be set in the status dictionary.

V_m        double - Membrane potential in mV
E_L        double - Leak reversal potential in mV.
C_m        double - Capacity of the membrane in pF
g_L        double - Leak conductance in nS;
I_e        double - Constant input current in pA.
a          double - Sigmoid function: theta_syn in mV
b          double - Sigmoid function: k_syn
g_ex       double - Gap junction conductance g_gap

Sends: CurrentEvent

Receives: CurrentEvent, DataLoggingRequest

Author: P. Martinez-Cañada (pablomc@ugr.es)

SeeAlso: parvo_neuron

References:

[1] Martinez-Cañada, P., Morillas, C., Pelayo, F. (2018). A Neuronal Network Model
of the Primate Visual System: Color Mechanisms in the Retina, LGN and V1. In
International Journal of Neural Systems. Accepted for publication.

[2] Martinez-Cañada, P., Morillas, C., Pelayo, F. (2017). A Conductance-Based
Neuronal Network Model for Color Coding in the Primate Foveal Retina. In IWINAC
2017

*/

/**
 * Function computing right-hand side of ODE for GSL solver.
 * @note Must be declared here so we can befriend it in class.
 * @note Must have C-linkage for passing to GSL. Internally, it is
 *       a first-class C++ function, but cannot be a member function
 *       because of the C-linkage.
 * @note No point in declaring it inline, since it is called
 *       through a function pointer.
 * @param void* Pointer to model neuron instance.
 */
extern "C" int AII_amacrine_dynamics( double, const double*, double*, void* );

/**
 * Non-spiking parvo neuron
 */
class AII_amacrine : public nest::Archiving_Node
{
public:
  // Class Constructor and Destructor
  AII_amacrine();
  AII_amacrine( const AII_amacrine& );
  ~AII_amacrine();

  // to have proper instances on each parallel process and to be able to accept
  // DataLoggingRequests from the Multimeter
  bool has_proxies()    const { return false; }
  bool local_receiver() const { return true;  }

  /**
   * Import sets of overloaded virtual functions.
   * This is necessary to ensure proper overload and overriding resolution.
   * @see http://www.gotw.ca/gotw/005.htm.
   */
  using nest::Node::handle;
  using nest::Node::handles_test_event;

  /**
   * Used to validate that we can send Event to desired target:port.
   */
  nest::port send_test_event( nest::Node&, nest::port, nest::synindex, bool );

  /**
   * @defgroup mynest_handle Functions handling incoming events.
   * We tell nest that we can handle incoming events of various types by
   * defining @c handle() and @c connect_sender() for the given event.
   * @{
   */
  void handle( nest::CurrentEvent& );       //! accept input current
  void handle( nest::DataLoggingRequest& ); //! allow recording with multimeter

  nest::port handles_test_event( nest::CurrentEvent&, nest::port );
  nest::port handles_test_event( nest::DataLoggingRequest&, nest::port );
  /** @} */

  void get_status( DictionaryDatum& ) const;
  void set_status( const DictionaryDatum& );

private:
  //! Reset parameters and state of neuron.

  //! Reset state of neuron.
  void init_state_( const Node& proto );

  //! Reset internal buffers of neuron.
  void init_buffers_();

  //! Initialize auxiliary quantities, leave parameters and state untouched.
  void calibrate();

  //! Take neuron through given time interval
  void update( nest::Time const&, const long, const long );

  // The next two classes need to be friends to access the State_ class/member
  friend class nest::RecordablesMap< AII_amacrine >;
  friend class nest::UniversalDataLogger< AII_amacrine >;

  // make dynamics function quasi-member
  friend int AII_amacrine_dynamics( double, const double*, double*, void* );

  /**
   * Free parameters of the neuron.
   *
   * These are the parameters that can be set by the user through @c SetStatus.
   * They are initialized from the model prototype when the node is created.
   * Parameters do not change during calls to @c update() and are not reset by
   * @c ResetNetwork.
   *
   * @note Parameters_ need neither copy constructor nor @c operator=(), since
   *       all its members are copied properly by the default copy constructor
   *       and assignment operator. Important:
   *       - If Parameters_ contained @c Time members, you need to define the
   *         assignment operator to recalibrate all members of type @c Time .
   *         You may also want to define the assignment operator.
   *       - If Parameters_ contained members that cannot copy themselves, such
   *         as C-style arrays, you need to define the copy constructor and
   *         assignment operator to copy those members.
   */
  struct Parameters_
  {
      double g_L;      //!< Leak Conductance in nS
      double C_m;      //!< Membrane Capacitance in pF
      double E_L;      //!< Leak reversal Potential (aka resting potential) in mV
      double I_e;      //!< Constant Current in pA
      double a;        //!< Sigmoid function: theta_syn in mV
      double b;        //!< Sigmoid function: k_syn
      double g_ex;     //!< Gap junction conductance g_gap

    //! Initialize parameters to their default values.
    Parameters_();

    //! Store parameter values in dictionary.
    void get( DictionaryDatum& ) const;

    //! Set parameter values from dictionary.
    void set( const DictionaryDatum& );
  };

  /**
   * Dynamic state of the neuron.
   *
   * These are the state variables that are advanced in time by calls to
   * @c update(). In many models, some or all of them can be set by the user
   * through @c SetStatus. The state variables are initialized from the model
   * prototype when the node is created. State variables are reset by @c
   * ResetNetwork.
   *
   * @note State_ need neither copy constructor nor @c operator=(), since
   *       all its members are copied properly by the default copy constructor
   *       and assignment operator. Important:
   *       - If State_ contained @c Time members, you need to define the
   *         assignment operator to recalibrate all members of type @c Time .
   *         You may also want to define the assignment operator.
   *       - If State_ contained members that cannot copy themselves, such
   *         as C-style arrays, you need to define the copy constructor and
   *         assignment operator to copy those members.
   */
  struct State_
  {
      enum StateVecElems
      {
        V_m = 0,
        STATE_VEC_SIZE
      };

      //! state vector, must be C-array for GSL solver
      double y[ STATE_VEC_SIZE ];

    /**
     * Construct new default State_ instance based on values in Parameters_.
     * This c'tor is called by the no-argument c'tor of the neuron model. It
     * takes a reference to the parameters instance of the model, so that the
     * state can be initialized in accordance with parameters, e.g.,
     * initializing the membrane potential with the resting potential.
     */
    State_( const Parameters_& );

    /** Store state values in dictionary. */
    void get( DictionaryDatum& ) const;

    /**
     * Set membrane potential from dictionary.
     * @note Receives Parameters_ so it can test that the new membrane potential
     *       is below threshold.
     */
    void set( const DictionaryDatum&, const Parameters_& );
  };

  /**
   * Buffers of the neuron.
   * Ususally buffers for incoming spikes and data logged for analog recorders.
   * Buffers must be initialized by @c init_buffers_(), which is called before
   * @c calibrate() on the first call to @c Simulate after the start of NEST,
   * ResetKernel or ResetNetwork.
   * @node Buffers_ needs neither constructor, copy constructor or assignment
   *       operator, since it is initialized by @c init_nodes_(). If Buffers_
   *       has members that cannot destroy themselves, Buffers_ will need a
   *       destructor.
   */
  struct Buffers_
  {
    Buffers_( AII_amacrine& );
    Buffers_( const Buffers_&, AII_amacrine& );

    nest::RingBuffer currents_exc;

    //! Logger for all analog data
    nest::UniversalDataLogger< AII_amacrine > logger_;

    /* GSL ODE stuff */
    gsl_odeiv_step* s_;    //!< stepping function
    gsl_odeiv_control* c_; //!< adaptive stepsize control function
    gsl_odeiv_evolve* e_;  //!< evolution function
    gsl_odeiv_system sys_; //!< struct describing system

    // IntergrationStep_ should be reset with the neuron on ResetNetwork,
    // but remain unchanged during calibration. Since it is initialized with
    // step_, and the resolution cannot change after nodes have been created,
    // it is safe to place both here.
    double step_;            //!< step size in ms
    double IntegrationStep_; //!< current integration time step, updated by GSL
    /**
     * Input current injected by CurrentEvent.
     * This variable is used to transport the current applied into the
     * _dynamics function computing the derivative of the state vector.
     * It must be a part of Buffers_, since it is initialized once before
     * the first simulation, but not modified before later Simulate calls.
     */
    double I_stim_exc;

  };

  /**
   * Internal variables of the neuron.
   * These variables must be initialized by @c calibrate, which is called before
   * the first call to @c update() upon each call to @c Simulate.
   * @node Variables_ needs neither constructor, copy constructor or assignment
   *       operator, since it is initialized by @c calibrate(). If Variables_
   *       has members that cannot destroy themselves, Variables_ will need a
   *       destructor.
   */
  struct Variables_
  {
        double g_gap_weight;
  };

  /**
   * @defgroup Access functions for UniversalDataLogger.
   * @{
   */
  //! Read out the real membrane potential
  double
  get_V_m_() const
  {
    return S_.y[State_::V_m];
  }
  /** @} */

  /**
   * @defgroup Member variables of neuron model.
   * Each model neuron should have precisely the following four data members,
   * which are one instance each of the parameters, state, buffers and variables
   * structures. Experience indicates that the state and variables member should
   * be next to each other to achieve good efficiency (caching).
   * @note Devices require one additional data member, an instance of the @c
   *       Device child class they belong to.
   * @{
   */
  Parameters_ P_; //!< Free parameters.
  State_ S_;      //!< Dynamic state.
  Variables_ V_;  //!< Internal Variables
  Buffers_ B_;    //!< Buffers.

  //! Mapping of recordables names to access functions
  static nest::RecordablesMap< AII_amacrine > recordablesMap_;

  /** @} */

};

inline nest::port
mynest::AII_amacrine::send_test_event( nest::Node& target,
  nest::port receptor_type,
  nest::synindex,
  bool )
{
  // You should usually not change the code in this function.
  // It confirms that the target of connection @c c accepts @c CurrentEvent on
  // the given @c receptor_type.
  nest::CurrentEvent e;
  e.set_sender( *this );
  return target.handles_test_event( e, receptor_type );
}

inline nest::port
mynest::AII_amacrine::handles_test_event( nest::CurrentEvent&,
  nest::port receptor_type )
{
  // You should usually not change the code in this function.
  // It confirms to the connection management system that we are able
  // to handle @c CurrentEvent on port 0. You need to extend the function
  // if you want to differentiate between input ports.
  if ( receptor_type != 0 )
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  return 0;
}

inline nest::port
mynest::AII_amacrine::handles_test_event( nest::DataLoggingRequest& dlr,
  nest::port receptor_type )
{
  // You should usually not change the code in this function.
  // It confirms to the connection management system that we are able
  // to handle @c DataLoggingRequest on port 0.
  // The function also tells the built-in UniversalDataLogger that this node
  // is recorded from and that it thus needs to collect data during simulation.
  if ( receptor_type != 0 )
    throw nest::UnknownReceptorType( receptor_type, get_name() );

  return B_.logger_.connect_logging_device( dlr, recordablesMap_ );
}

inline void
AII_amacrine::get_status( DictionaryDatum& d ) const
{
  // get our own parameter and state data
  P_.get( d );
  S_.get( d );

  // get information managed by parent class
  Archiving_Node::get_status( d );

  ( *d )[ nest::names::recordables ] = recordablesMap_.get_list();
}

inline void
AII_amacrine::set_status( const DictionaryDatum& d )
{
  Parameters_ ptmp = P_; // temporary copy in case of errors
  ptmp.set( d );         // throws if BadProperty
  State_ stmp = S_;      // temporary copy in case of errors
  stmp.set( d, ptmp );   // throws if BadProperty

  // We now know that (ptmp, stmp) are consistent. We do not
  // write them back to (P_, S_) before we are also sure that
  // the properties to be set in the parent class are internally
  // consistent.
  Archiving_Node::set_status( d );

  // if we get here, temporaries contain consistent set of properties
  P_ = ptmp;
  S_ = stmp;
}

} // namespace

#endif /* #ifndef AII_AMACRINE_H */
